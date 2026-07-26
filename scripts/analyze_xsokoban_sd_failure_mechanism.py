#!/usr/bin/env python3
"""Diagnose why the current GFT-SD-MCTS policy fails on XSokoban first level.

This script focuses on the model-side mechanism, not on benchmark reporting.
It replays the saved trace prefix from the existing max_steps=1000 failure log,
recomputes GFT resource values and candidate-action diagnostics at each saved
root, and writes a compact JSON/Markdown analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPERIMENT_DIR = ROOT / "experiments" / "sokoban_sd_world_model"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from src.sokoban import SokobanGFTSDModel, SokobanPushAction, load_boxoban_levels
from src.sokoban.env import ACTION_NAMES, render_state

from evaluate_sokoban_gft_sdmcts import (  # noqa: E402
    GFTTvaCheckpoint,
    discounted_progress_gft_sd_value,
    load_tva_checkpoint,
    predict_gft_tva_details,
)


def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "inf"
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def build_model(level, config: dict[str, Any]) -> SokobanGFTSDModel:
    return SokobanGFTSDModel(
        level,
        transport_cost_weights={key: float(value) for key, value in config["gft_transport_cost_weights"].items()},
        soft_deadlock_weights={key: float(value) for key, value in config["gft_soft_deadlock_weights"].items()},
        commitment_beta=float(config["gft_commitment_beta"]),
        sinkhorn_iterations=int(config["gft_sinkhorn_iterations"]),
    )


def resource_values(model: SokobanGFTSDModel, state) -> dict[str, float]:
    return {row.code: float(row.value) for row in model.gft_state(state, active_only=True).resources}


def gft_leaf_value(model: SokobanGFTSDModel, state, config: dict[str, Any]) -> float:
    return discounted_progress_gft_sd_value(
        model,
        state,
        gamma=float(config["gamma"]),
        goal_value=float(config["goal_value"]),
        push_cost=float(config["push_cost"]),
        deadend_value=float(config["deadend_value"]),
        weights={key: float(value) for key, value in config["gft_sd_leaf_weights"].items()},
    )


def root_transport_phase_eligible(model: SokobanGFTSDModel, state, config: dict[str, Any]) -> bool:
    features = model.gft_tva_feature_dict(state)["features"]
    progress = float(features.get("P", math.inf))
    commitment = float(features.get("C", 0.0))
    if not math.isfinite(progress):
        return False
    high = (
        float(config.get("tva_transport_phase_high_progress_min", -1.0)) >= 0.0
        and progress >= float(config.get("tva_transport_phase_high_progress_min", -1.0))
    )
    low = (
        float(config.get("tva_transport_phase_low_progress_max", -1.0)) >= 0.0
        and progress <= float(config.get("tva_transport_phase_low_progress_max", -1.0))
        and commitment >= float(config.get("tva_transport_phase_commitment_min", 0.25))
    )
    return bool(high or low)


def state_gated_alpha(
    model: SokobanGFTSDModel,
    state,
    step: int,
    root_eligible: bool,
    vsd: float,
    prediction: float,
    details: dict[str, float],
    config: dict[str, Any],
) -> tuple[float, str]:
    alpha = float(config.get("tva_state_gated_default_blend", 0.0))
    threshold = float(config.get("tva_state_gated_ood_z_threshold", -1.0))
    if threshold > 0.0 and float(details.get("feature_max_abs_z", 0.0)) > threshold:
        return float(config.get("tva_state_gated_ood_blend", 0.0)), "ood"

    transport_blend = float(config.get("tva_transport_phase_blend", -1.0))
    if transport_blend >= 0.0:
        features = model.gft_tva_feature_dict(state)["features"]
        progress = float(features.get("P", math.inf))
        commitment = float(features.get("C", 0.0))
        max_step = int(config.get("tva_transport_phase_max_closed_loop_step", -1))
        within_prefix = max_step < 0 or step <= max_step
        root_allowed = (not bool(config.get("tva_transport_phase_require_root_eligible", False))) or root_eligible
        high = (
            float(config.get("tva_transport_phase_high_progress_min", -1.0)) >= 0.0
            and progress >= float(config.get("tva_transport_phase_high_progress_min", -1.0))
        )
        low = (
            float(config.get("tva_transport_phase_low_progress_max", -1.0)) >= 0.0
            and progress <= float(config.get("tva_transport_phase_low_progress_max", -1.0))
            and commitment >= float(config.get("tva_transport_phase_commitment_min", 0.25))
        )
        if within_prefix and root_allowed and math.isfinite(progress) and (high or low):
            return transport_blend, "transport_phase"

    delta = float(prediction) - float(vsd)
    risk = float(details.get("risk_probability", 0.0))
    if (
        risk <= float(config.get("tva_state_gated_positive_risk_max", 0.3))
        and delta >= float(config.get("tva_state_gated_positive_delta_min", 0.25))
    ):
        positive = float(config.get("tva_state_gated_positive_blend", -1.0))
        return (float(config.get("tva_value_blend", 1.0)) if positive < 0.0 else positive), "positive_delta"
    if (
        risk >= float(config.get("tva_state_gated_negative_risk_min", 0.95))
        and delta <= -float(config.get("tva_state_gated_negative_delta_min", 1.0))
    ):
        return float(config.get("tva_state_gated_negative_blend", 0.0)), "negative_risk"
    return alpha, "default"


def action_key(action: SokobanPushAction | dict[str, Any]) -> tuple[int, int]:
    if isinstance(action, SokobanPushAction):
        return int(action.box_index), int(action.direction)
    return int(action["box_index"]), int(action["direction"])


def action_label(action: SokobanPushAction | dict[str, Any]) -> str:
    box, direction = action_key(action)
    return f"b{box}:{ACTION_NAMES[direction]}"


def rank_of(rows: list[dict[str, Any]], selected_key: tuple[int, int], key: str, reverse: bool = True) -> int | None:
    ordered = sorted(rows, key=lambda row: (row[key], row.get("prior", 0.0)), reverse=reverse)
    for idx, row in enumerate(ordered, start=1):
        if tuple(row["action_key"]) == selected_key:
            return idx
    return None


def candidate_diagnostics(
    model: SokobanGFTSDModel,
    state,
    step: int,
    selected: SokobanPushAction,
    visited_keys: set,
    config: dict[str, Any],
    checkpoint: GFTTvaCheckpoint | None,
) -> list[dict[str, Any]]:
    candidates = []
    for action in model.legal_flows(state):
        child = model.step_grid_flow(state, action)
        if model.is_tier_a_deadlocked(child):
            continue
        candidates.append((action, child))

    rows = model.gft_action_prior_rows(
        state,
        candidates,
        visited_keys=visited_keys,
        weights={key: float(value) for key, value in config["gft_prior_weights"].items()},
        temperature=float(config["gft_prior_temperature"]),
        uniform_mix=float(config["gft_prior_uniform_mix"]),
    )
    root_eligible = root_transport_phase_eligible(model, state, config)
    out: list[dict[str, Any]] = []
    for prior_row, (action, child) in zip(rows, candidates):
        vsd = gft_leaf_value(model, child, config)
        if checkpoint is not None:
            details = predict_gft_tva_details(checkpoint, model, child)
            prediction = float(details["prediction"])
            alpha, alpha_reason = state_gated_alpha(
                model,
                child,
                step,
                root_eligible,
                vsd,
                prediction,
                details,
                config,
            )
            leaf = (1.0 - alpha) * vsd + alpha * prediction
        else:
            details = {}
            prediction = math.nan
            alpha = 0.0
            alpha_reason = "no_checkpoint"
            leaf = vsd
        resources = resource_values(model, child)
        components = prior_row["components"]
        out.append(
            {
                "action": {
                    "box_index": int(action.box_index),
                    "direction": int(action.direction),
                    "direction_name": ACTION_NAMES[action.direction],
                    "flat_index": int(action.flat_index),
                },
                "action_key": [int(action.box_index), int(action.direction)],
                "selected": action_key(action) == action_key(selected),
                "prior_score": float(prior_row["score"]),
                "prior": float(prior_row["prior"]),
                "prior_components": components,
                "child_resources": resources,
                "child_vsd": float(vsd),
                "child_tva_prediction": _safe_float(prediction),
                "child_tva_risk": _safe_float(details.get("risk_probability")),
                "child_feature_max_abs_z": _safe_float(details.get("feature_max_abs_z")),
                "child_leaf_value": float(leaf),
                "child_alpha": float(alpha),
                "child_alpha_reason": alpha_reason,
                "boxes_on_goals": int(model.boxes_on_goals(child)),
                "tier_a": model.tier_a_status(child),
            }
        )
    return out


def summarize_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    child_counts = [int(row["child_count"]) for row in rows]
    visit_shares = [float(row["selected_visit_share"]) for row in rows if row.get("selected_visit_share") is not None]
    selected_prior_ranks = [row["selected_prior_rank"] for row in rows if row["selected_prior_rank"] is not None]
    selected_leaf_ranks = [row["selected_leaf_rank"] for row in rows if row["selected_leaf_rank"] is not None]
    selected_actions = [row["selected_action_label"] for row in rows]
    selected_boxes = [row["selected_action"]["box_index"] for row in rows]
    resources0 = rows[0]["root_resources"] if rows else {}
    resources_last = rows[-1]["after_resources"] if rows else {}
    p_deltas = [row["selected_prior_components"]["progress_delta"] for row in rows]
    q_deltas = [row["selected_prior_components"]["ordering_delta"] for row in rows]
    d_deltas = [row["selected_prior_components"]["deadlock_delta"] for row in rows]
    z_values = [
        float(row["selected_feature_max_abs_z"])
        for row in rows
        if row.get("selected_feature_max_abs_z") is not None
        and math.isfinite(float(row["selected_feature_max_abs_z"]))
    ]
    alpha_reasons = [row["selected_alpha_reason"] for row in rows]
    return {
        "trace_steps": len(rows),
        "single_child_steps": sum(1 for value in child_counts if value == 1),
        "child_count_median": median(child_counts) if child_counts else None,
        "selected_visit_share_mean": mean(visit_shares) if visit_shares else None,
        "selected_visit_share_median": median(visit_shares) if visit_shares else None,
        "selected_action_counts": dict(Counter(selected_actions)),
        "selected_box_counts": dict(Counter(str(value) for value in selected_boxes)),
        "selected_prior_rank_counts": dict(Counter(str(value) for value in selected_prior_ranks)),
        "selected_leaf_rank_counts": dict(Counter(str(value) for value in selected_leaf_ranks)),
        "selected_feature_max_abs_z": {
            "min": min(z_values) if z_values else None,
            "mean": mean(z_values) if z_values else None,
            "max": max(z_values) if z_values else None,
        },
        "selected_alpha_reason_counts": dict(Counter(alpha_reasons)),
        "progress_delta_sum_prefix": sum(float(value) for value in p_deltas),
        "ordering_delta_sum_prefix": sum(float(value) for value in q_deltas),
        "deadlock_delta_sum_prefix": sum(float(value) for value in d_deltas),
        "resource_change_prefix": {
            key: float(resources_last.get(key, 0.0) - resources0.get(key, 0.0))
            for key in ("P", "M", "A", "K", "D", "Q", "C")
            if key in resources0 and key in resources_last
        },
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["prefix_summary"]
    cfg = payload["config_extract"]
    first = payload["trace_rows"][0]
    last = payload["trace_rows"][-1]
    final = payload["final_failure"]

    lines = [
        "# XSokoban first 第 1 关的 SD 模型失败机制分析",
        "",
        "## 核心结论",
        "",
        (
            "当前失败主要不是由 `max_steps` 或单次 MCTS 展开预算不足造成，而是由当前 SD/GFT/TVA 决策层的结构性偏差造成："
            "模型把多箱目标房间问题压缩为局部运输进展最大化问题，缺少可执行的 box-goal/room-order 长程承诺。"
        ),
        (
            "在保存的前 20 步 trace 中，规划很快进入低分支走廊状态，"
            f"`child_count=1` 的 root 有 `{summary['single_child_steps']}/{summary['trace_steps']}` 个；"
            f"被选动作访问占比中位数为 `{_fmt(summary['selected_visit_share_median'])}`。"
            "这说明 4096 次 MCTS 展开多数时候没有在多个战略分支之间比较，而是在窄通道中反复确认同一个后继。"
        ),
        (
            f"前缀中动作 `{max(summary['selected_action_counts'], key=summary['selected_action_counts'].get)}` "
            f"出现 `{max(summary['selected_action_counts'].values())}` 次，"
            "表现为把同一运输方向持续推深。该行为能降低短期运输成本，但没有保证目标房间的填充顺序。"
        ),
        (
            f"同时，被选后继按一阶 leaf value 排名第一的步数为 "
            f"`{summary['selected_leaf_rank_counts'].get('1', 0)}/{summary['trace_steps']}`。"
            "这说明失败不是因为 MCTS 没有服从当前价值函数，而是当前价值函数本身把错误的局部运输前缀评估得足够好。"
        ),
        "",
        "## 当前配置暴露的关键缺口",
        "",
        (
            "1. **动作先验几乎只奖励运输进展**。本次配置中 root prior 权重为 "
            f"`progress={_fmt(cfg['prior_weights']['progress'])}`、`goal={_fmt(cfg['prior_weights']['goal'])}`、"
            f"`ordering={_fmt(cfg['prior_weights']['ordering'])}`、`commitment={_fmt(cfg['prior_weights']['commitment'])}`，"
            f"但 `mobility/access/capacity/deadlock` 权重均为 `{_fmt(0.0)}`。"
            "因此，模型能计算 D/K/A/M 等资源，却不会在动作先验中强烈惩罚软死锁、容量压力或可达性恶化。"
        ),
        (
            "2. **叶值同样弱化房间顺序风险**。GFT leaf 只给 `capacity/order/commitment` 各很小的修正，"
            "且 `deadlock` leaf 权重为 0；目标房间顺序错误在成为硬死锁之前，不足以覆盖 progress 和 goal 的短期收益。"
        ),
        (
            "3. **TVA 校正没有承担 OOD 保护**。TVA 的 state-gated 默认 blend 为 0，"
            "只在早期 transport phase 或低风险正修正时介入；同时 OOD z-threshold 被关闭。"
            f"保存 trace 中被选后继的 TVA 特征 `max_abs_z` 均值为 "
            f"`{_fmt(summary['selected_feature_max_abs_z']['mean'])}`，最大值为 "
            f"`{_fmt(summary['selected_feature_max_abs_z']['max'])}`；"
            "这些状态明显超出 TVA 训练/审计尺度，但当前配置不会据此触发明确的 OOD 降级或保守策略。"
        ),
        (
            "4. **revisit 惩罚无法处理单调错误承诺**。重复向右推进同一箱子会产生新 box layout，"
            "不是 canonical revisit；因此 revisit 只能惩罚回环，不能阻止“持续向错误目标房间深处推进”的策略。"
        ),
        "",
        "## 前缀资源变化",
        "",
        "| 指标 | trace 初始 root | 第 20 步后 | 前缀变化 |",
        "|---|---:|---:|---:|",
    ]
    for key in ("P", "M", "A", "K", "D", "Q", "C"):
        before = first["root_resources"][key]
        after = last["after_resources"][key]
        lines.append(f"| `{key}` | {_fmt(before)} | {_fmt(after)} | {_fmt(after - before)} |")
    lines.extend(
        [
            "",
            "这里的关键现象是：`P` 明显下降，说明模型确实在制造运输进展；但 `Q` 从 0 升高，"
            "`D` 没有显著下降，`C` 仍很低。这是典型的“局部距离进展成立，但目标房间承诺和顺序控制失败”。",
        "",
        "TVA 介入方式也支持这一判断：前缀中 alpha 触发计数为 "
        f"`{summary['selected_alpha_reason_counts']}`。前 6 步主要由 transport phase 规则介入，"
        "之后大多回到默认 alpha=0；少数 positive_delta 介入发生在已经出现顺序债务的状态上，不能提供房间顺序保护。",
        "",
        "## 代表性 root 诊断",
            "",
            "| step | selected | child_count | visit_share | prior_rank | leaf_rank | progress_delta | ordering_delta | deadlock_delta | leaf_alpha | alpha_reason |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["trace_rows"]:
        comp = row["selected_prior_components"]
        lines.append(
            "| "
            f"{row['step']} | {row['selected_action_label']} | {row['child_count']} | "
            f"{_fmt(row['selected_visit_share'])} | {_fmt(row['selected_prior_rank'])} | "
            f"{_fmt(row['selected_leaf_rank'])} | {_fmt(comp['progress_delta'])} | "
            f"{_fmt(comp['ordering_delta'])} | {_fmt(comp['deadlock_delta'])} | "
            f"{_fmt(row['selected_alpha'])} | {row['selected_alpha_reason']} |"
        )
    lines.extend(
        [
            "",
            "## 为什么最终 1000 步仍失败",
            "",
            (
                f"最终状态仍为 `{final['status']}`，只完成 `{final['boxes_on_goals']}` 个目标，"
                f"最终 `P={_fmt(final['resources']['P'])}`、`M={_fmt(final['resources']['M'])}`、"
                f"`A={_fmt(final['resources']['A'])}`、`D={_fmt(final['resources']['D'])}`、"
                f"`Q={_fmt(final['resources']['Q'])}`、`C={_fmt(final['resources']['C'])}`。"
            ),
            (
                "这说明模型没有走入硬死锁，而是停留在“仍有合法 push、但目标房间填充顺序已经不受控制”的状态。"
                "增大 `max_steps` 只能给这种局部策略更多执行时间，不能自动生成多箱房间的承诺结构。"
            ),
            "",
            "## 直接方法含义",
            "",
            (
                "若继续推进 XSokoban，优先修复点不应是单纯增大闭环步数，而是让 SD 层产生可消费的长程承诺："
                "显式 box-goal/room-slot 绑定、目标房间入口到深处的填充顺序、对错误深推的早期惩罚，以及 XSokoban 分布上的 TVA/OOD 校准。"
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--failure-log",
        default="results/sokoban_sd_world_model/xsokoban_first40_bestconfig_exact_l001_maxsteps1000_seed0.json",
    )
    parser.add_argument("--out-json", default="results/sokoban_sd_world_model/xsokoban_sd_failure_mechanism_analysis.json")
    parser.add_argument("--out-md", default="docs/xsokoban_sd_failure_mechanism_analysis.md")
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    config = failure["config"]
    framework = failure.get("framework_contract", {})
    level = load_boxoban_levels(ROOT / config["levels"], max_levels=int(config["level_offset"]) + 1)[
        int(config["level_offset"])
    ]
    model = build_model(level, config)
    tva_path = framework.get("tva_checkpoint")
    checkpoint = load_tva_checkpoint(ROOT / tva_path) if tva_path else None
    if checkpoint is not None and not isinstance(checkpoint, GFTTvaCheckpoint):
        checkpoint = None

    state = level.initial_state
    visited_keys = {model.canonical_key(state)}
    trace_rows: list[dict[str, Any]] = []
    saved_trace = failure["levels"][0]["trace"]
    for trace in saved_trace:
        root_resources = resource_values(model, state)
        selected = SokobanPushAction(
            box_index=int(trace["action"]["box_index"]),
            direction=int(trace["action"]["direction"]),
        )
        candidates = candidate_diagnostics(model, state, int(trace["step"]), selected, visited_keys, config, checkpoint)
        selected_rows = [row for row in candidates if row["selected"]]
        if not selected_rows:
            raise RuntimeError(f"Saved selected action is not legal at trace step {trace['step']}.")
        selected_row = selected_rows[0]
        selected_key = tuple(selected_row["action_key"])
        top_children = trace.get("top_children", [])
        selected_visit_share = None
        selected_mean_value = None
        for child in top_children:
            if action_key(child["action"]) == selected_key:
                selected_visit_share = float(child["visits"]) / float(trace["root_visits"])
                selected_mean_value = float(child["mean_value"])
                break
        next_state = model.step_grid_flow(state, selected)
        after_resources = resource_values(model, next_state)
        trace_rows.append(
            {
                "step": int(trace["step"]),
                "selected_action": selected_row["action"],
                "selected_action_label": action_label(selected),
                "child_count": int(trace["child_count"]),
                "root_visits": int(trace["root_visits"]),
                "root_value": float(trace["root_value"]),
                "selected_visit_share": selected_visit_share,
                "selected_mean_value": selected_mean_value,
                "selected_prior_rank": rank_of(candidates, selected_key, "prior_score", reverse=True),
                "selected_prior_probability_rank": rank_of(candidates, selected_key, "prior", reverse=True),
                "selected_leaf_rank": rank_of(candidates, selected_key, "child_leaf_value", reverse=True),
                "selected_vsd_rank": rank_of(candidates, selected_key, "child_vsd", reverse=True),
                "selected_prior_components": selected_row["prior_components"],
                "selected_child_resources": selected_row["child_resources"],
                "selected_child_leaf_value": selected_row["child_leaf_value"],
                "selected_child_vsd": selected_row["child_vsd"],
                "selected_child_tva_prediction": selected_row["child_tva_prediction"],
                "selected_child_tva_risk": selected_row["child_tva_risk"],
                "selected_feature_max_abs_z": selected_row["child_feature_max_abs_z"],
                "selected_alpha": selected_row["child_alpha"],
                "selected_alpha_reason": selected_row["child_alpha_reason"],
                "root_resources": root_resources,
                "after_resources": after_resources,
                "candidate_count_recomputed": len(candidates),
                "candidate_rows": sorted(
                    candidates,
                    key=lambda row: (row["child_leaf_value"], row["prior_score"]),
                    reverse=True,
                ),
                "state_before": render_state(level, state),
                "state_after": render_state(level, next_state),
            }
        )
        state = next_state
        visited_keys.add(model.canonical_key(state))

    final_level = failure["levels"][0]
    final_resources = {key: float(value["value"]) for key, value in final_level["final_gft_state"]["resources"].items()}
    payload = {
        "contract": {
            "analysis": "xsokoban_sd_failure_mechanism_v1",
            "failure_log": _rel(failure_path),
            "tva_checkpoint": tva_path,
            "level_id": level.level_id,
            "trace_scope": "saved_trace_prefix_only",
        },
        "config_extract": {
            "horizon": int(config["horizon"]),
            "budget": int(config["budget"]),
            "max_steps": int(config["max_steps"]),
            "leaf_value": config["leaf_value"],
            "select_by": config["select_by"],
            "prior_weights": config["gft_prior_weights"],
            "leaf_weights": config["gft_sd_leaf_weights"],
            "tva_state_gated_default_blend": config["tva_state_gated_default_blend"],
            "tva_transport_phase_blend": config["tva_transport_phase_blend"],
            "tva_transport_phase_max_closed_loop_step": config["tva_transport_phase_max_closed_loop_step"],
            "tva_state_gated_ood_z_threshold": config["tva_state_gated_ood_z_threshold"],
        },
        "prefix_summary": summarize_trace(trace_rows),
        "trace_rows": trace_rows,
        "final_failure": {
            "status": final_level["status"],
            "steps_executed": int(final_level["steps_executed"]),
            "boxes_on_goals": int(final_level["final_features"]["boxes_on_goals"]),
            "legal_push_count": int(final_level["final_features"]["legal_push_count"]),
            "viable_push_count": int(final_level["final_features"]["viable_push_count"]),
            "deadlocked": bool(final_level["final_features"]["deadlocked"]),
            "resources": final_resources,
        },
    }

    out_json = ROOT / args.out_json
    out_md = ROOT / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(out_md, payload)
    print(json.dumps({"out_json": _rel(out_json), "out_md": _rel(out_md)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
