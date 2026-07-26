#!/usr/bin/env python3
"""A1/A2 diagnostics for the room-aware XSokoban hard-instance SD line.

This script replays the saved XSokoban first-level failure trace.  It does not
run a new long closed-loop search.  The goal is to audit two isolated changes:

* A1: suppress positive TVA corrections when the candidate state is OOD.
* A2: expose goal-room, slot-frontier, and uncommitted room-entry signals.
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

from src.sokoban import SokobanPushAction, SokobanRoomGFTSDModel, load_boxoban_levels
from src.sokoban.env import ACTION_NAMES

from analyze_xsokoban_sd_failure_mechanism import (  # noqa: E402
    _fmt,
    _rel,
    _safe_float,
    action_key,
    action_label,
    gft_leaf_value,
    rank_of,
    resource_values,
    root_transport_phase_eligible,
    state_gated_alpha,
)
from evaluate_sokoban_gft_sdmcts import (  # noqa: E402
    GFTTvaCheckpoint,
    load_tva_checkpoint,
    predict_gft_tva_details,
)


def build_room_model(level, config: dict[str, Any]) -> SokobanRoomGFTSDModel:
    return SokobanRoomGFTSDModel(
        level,
        transport_cost_weights={key: float(value) for key, value in config["gft_transport_cost_weights"].items()},
        soft_deadlock_weights={key: float(value) for key, value in config["gft_soft_deadlock_weights"].items()},
        commitment_beta=float(config["gft_commitment_beta"]),
        sinkhorn_iterations=int(config["gft_sinkhorn_iterations"]),
    )


def _delta_totals(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    b = before["totals"]
    a = after["totals"]
    return {key: int(a.get(key, 0) - b.get(key, 0)) for key in sorted(set(b) | set(a))}


def safe_tva_leaf(
    *,
    vsd: float,
    prediction: float,
    current_alpha: float,
    feature_z: float | None,
    ood_z_threshold: float,
) -> tuple[float, str, float, bool]:
    """Apply the A1 rule: OOD states may not receive positive TVA correction."""

    positive = math.isfinite(prediction) and prediction > vsd
    ood = feature_z is not None and math.isfinite(feature_z) and feature_z > ood_z_threshold
    if ood and positive:
        return float(vsd), "ood_positive_suppressed", 0.0, True
    leaf = (1.0 - current_alpha) * vsd + current_alpha * prediction
    return float(leaf), "current_rule_retained", float(current_alpha), False


def candidate_diagnostics(
    model: SokobanRoomGFTSDModel,
    state,
    step: int,
    selected: SokobanPushAction,
    visited_keys: set,
    config: dict[str, Any],
    checkpoint: GFTTvaCheckpoint | None,
    ood_z_threshold: float,
) -> list[dict[str, Any]]:
    candidates = []
    for action in model.legal_flows(state):
        child = model.step_grid_flow(state, action)
        if model.is_tier_a_deadlocked(child):
            continue
        candidates.append((action, child))

    prior_rows = model.gft_action_prior_rows(
        state,
        candidates,
        visited_keys=visited_keys,
        weights={key: float(value) for key, value in config["gft_prior_weights"].items()},
        temperature=float(config["gft_prior_temperature"]),
        uniform_mix=float(config["gft_prior_uniform_mix"]),
    )
    root_eligible = root_transport_phase_eligible(model, state, config)
    root_resources = resource_values(model, state)
    out: list[dict[str, Any]] = []
    for prior_row, (action, child) in zip(prior_rows, candidates):
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
            current_leaf = (1.0 - alpha) * vsd + alpha * prediction
        else:
            details = {}
            prediction = math.nan
            alpha = 0.0
            alpha_reason = "no_checkpoint"
            current_leaf = vsd

        feature_z = _safe_float(details.get("feature_max_abs_z"))
        safe_leaf, safe_reason, safe_alpha, suppressed = safe_tva_leaf(
            vsd=vsd,
            prediction=prediction,
            current_alpha=alpha,
            feature_z=feature_z,
            ood_z_threshold=ood_z_threshold,
        )
        child_resources = resource_values(model, child)
        room_action = model.room_action_summary(state, action, child)
        semantic = model.room_semantic_options(state, action, child)
        p_down = child_resources.get("P", math.inf) < root_resources.get("P", -math.inf)
        q_up = child_resources.get("Q", -math.inf) > root_resources.get("Q", math.inf)
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
                "prior_components": prior_row["components"],
                "child_resources": child_resources,
                "child_vsd": float(vsd),
                "child_tva_prediction": _safe_float(prediction),
                "child_tva_risk": _safe_float(details.get("risk_probability")),
                "child_feature_max_abs_z": feature_z,
                "child_current_alpha": float(alpha),
                "child_current_alpha_reason": alpha_reason,
                "child_current_leaf_value": float(current_leaf),
                "child_safe_alpha": float(safe_alpha),
                "child_safe_leaf_value": float(safe_leaf),
                "child_safe_reason": safe_reason,
                "child_positive_tva_correction": bool(math.isfinite(prediction) and prediction > vsd),
                "child_ood_by_a1_threshold": bool(feature_z is not None and feature_z > ood_z_threshold),
                "child_positive_tva_on_ood_suppressed": bool(suppressed),
                "child_p_down_q_up": bool(p_down and q_up),
                "room_action": room_action,
                "room_delta_totals": _delta_totals(room_action["room_before"], room_action["room_after"]),
                "room_semantic_options": semantic,
                "tier_a": model.tier_a_status(child),
            }
        )
    return out


def summarize_a1(rows: list[dict[str, Any]], ood_z_threshold: float) -> dict[str, Any]:
    selected = [row["selected_candidate"] for row in rows]
    candidates = [cand for row in rows for cand in row["candidate_rows"]]

    def count(items: list[dict[str, Any]], key: str) -> int:
        return sum(1 for item in items if item.get(key))

    selected_z = [
        cand["child_feature_max_abs_z"]
        for cand in selected
        if cand.get("child_feature_max_abs_z") is not None
    ]
    active_suppressed = [
        cand
        for cand in selected
        if cand["child_positive_tva_on_ood_suppressed"] and cand["child_current_alpha"] > 0.0
    ]
    return {
        "ood_z_threshold": float(ood_z_threshold),
        "candidate_states": len(candidates),
        "candidate_ood_count": count(candidates, "child_ood_by_a1_threshold"),
        "candidate_positive_tva_count": count(candidates, "child_positive_tva_correction"),
        "candidate_positive_tva_on_ood_count": count(candidates, "child_positive_tva_on_ood_suppressed"),
        "selected_states": len(selected),
        "selected_ood_count": count(selected, "child_ood_by_a1_threshold"),
        "selected_positive_tva_count": count(selected, "child_positive_tva_correction"),
        "selected_positive_tva_on_ood_count": count(selected, "child_positive_tva_on_ood_suppressed"),
        "selected_active_positive_tva_on_ood_count": len(active_suppressed),
        "selected_alpha_reason_counts": dict(Counter(cand["child_current_alpha_reason"] for cand in selected)),
        "selected_safe_reason_counts": dict(Counter(cand["child_safe_reason"] for cand in selected)),
        "selected_feature_max_abs_z": {
            "min": min(selected_z) if selected_z else None,
            "mean": mean(selected_z) if selected_z else None,
            "max": max(selected_z) if selected_z else None,
        },
        "selected_leaf_value_delta_current_minus_safe_sum": sum(
            float(cand["child_current_leaf_value"] - cand["child_safe_leaf_value"])
            for cand in selected
        ),
    }


def summarize_a2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row["selected_candidate"] for row in rows]
    impact_counter = Counter(code for cand in selected for code in cand["room_action"]["impact_codes"])
    semantic_counts = [row["semantic_child_count_proxy_total"] for row in rows]
    physical_counts = [row["physical_child_count_log"] for row in rows]
    selected_semantic_counts = [cand["room_semantic_options"]["semantic_child_count_proxy"] for cand in selected]
    room_deltas = [cand["room_delta_totals"] for cand in selected]
    after_totals = [cand["room_action"]["room_after"]["totals"] for cand in selected]
    p_down_q_up = sum(1 for cand in selected if cand["child_p_down_q_up"])
    return {
        "trace_steps": len(rows),
        "single_physical_child_steps": sum(1 for value in physical_counts if value == 1),
        "physical_child_count_median": median(physical_counts) if physical_counts else None,
        "semantic_child_count_proxy_median": median(semantic_counts) if semantic_counts else None,
        "semantic_child_count_proxy_max": max(semantic_counts) if semantic_counts else None,
        "selected_semantic_child_count_proxy_median": (
            median(selected_semantic_counts) if selected_semantic_counts else None
        ),
        "impact_code_counts": dict(impact_counter),
        "frontier_fill_count": int(impact_counter.get("FILL_FRONTIER_SLOT", 0)),
        "nonfrontier_fill_count": int(impact_counter.get("FILL_NONFRONTIER_SLOT", 0)),
        "uncommitted_entry_transition_count": int(impact_counter.get("UNCOMMITTED_ENTRY", 0)),
        "uncommitted_deep_push_count": int(impact_counter.get("UNCOMMITTED_DEEP_PUSH", 0)),
        "room_order_debt_delta_sum": sum(max(0, int(delta.get("q_settled", 0))) for delta in room_deltas),
        "blocked_slot_delta_sum": sum(max(0, int(delta.get("k_blocked", 0))) for delta in room_deltas),
        "uncommitted_entry_after_sum": sum(int(total.get("uncommitted_entry", 0)) for total in after_totals),
        "uncommitted_in_room_after_sum": sum(int(total.get("uncommitted_in_room", 0)) for total in after_totals),
        "uncommitted_room_or_entry_after_sum": sum(
            int(total.get("uncommitted_room_or_entry", 0)) for total in after_totals
        ),
        "p_down_q_up_selected_steps": int(p_down_q_up),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    a1 = payload["a1_ood_safe_tva_summary"]
    a2 = payload["a2_room_summary"]
    static = payload["room_static_summary"]
    first_room = static["rooms"][0] if static["rooms"] else {"slots": [], "entries": []}
    trace0 = payload["trace_rows"][0]
    tracelast = payload["trace_rows"][-1]

    lines = [
        "# XSokoban first 第 1 关 Room-GFT A1/A2 诊断",
        "",
        "## 诊断结论",
        "",
        (
            "本轮建立了 `SokobanRoomGFTSDModel` 的只读派生诊断线。"
            "它继承当前 `SokobanGFTSDModel`，不修改 Boxoban 主线 planner、默认评估入口或既有结果文件。"
        ),
        (
            f"A1 诊断使用 `feature_max_abs_z>{_fmt(a1['ood_z_threshold'])}` 作为 OOD 阈值。"
            f"保存 trace 的 `{a1['selected_states']}` 个被选后继中，"
            f"OOD 状态为 `{a1['selected_ood_count']}` 个，"
            f"OOD 且 TVA 给出正向修正的状态为 `{a1['selected_positive_tva_on_ood_count']}` 个；"
            f"其中当前 alpha 实际大于 0 的正向 OOD 修正为 `{a1['selected_active_positive_tva_on_ood_count']}` 个。"
            "因此，A1 的直接作用是切断少量但危险的分布外正向补偿；它本身不足以生成新的房间填充策略。"
        ),
        (
            f"A2 诊断识别出 `{static['room_count']}` 个 goal room。第一个 room 含 "
            f"`{len(first_room['slots'])}` 个 slot、`{len(first_room['entries'])}` 个 entry/support cell。"
            f"保存 trace 中物理 `child_count=1` 的 root 为 "
            f"`{a2['single_physical_child_steps']}/{a2['trace_steps']}`；"
            f"语义 child 代理的中位数为 `{_fmt(a2['semantic_child_count_proxy_median'])}`、最大值为 "
            f"`{_fmt(a2['semantic_child_count_proxy_max'])}`。"
            "这说明 A3 需要把 room-slot commitment 真正并入 MCTS 节点；仅做只读诊断不会自动增加搜索分支。"
        ),
        "",
        "## 静态 room-slot 结构",
        "",
        "| room | entries | slots | frontier rule proxy |",
        "|---:|---:|---:|---|",
    ]
    for room in static["rooms"]:
        ordered = []
        for slot in room["slots"]:
            if slot["must_follow"]:
                ordered.append(f"{slot['cell']} waits {slot['must_follow']}")
        rule = "; ".join(ordered) if ordered else "no inferred precedence"
        lines.append(f"| {room['room_id']} | {len(room['entries'])} | {len(room['slots'])} | {rule} |")

    lines.extend(
        [
            "",
            "## A1：TVA OOD 正修正审计",
            "",
            "| 指标 | 数值 |",
            "|---|---:|",
            f"| candidate states | {a1['candidate_states']} |",
            f"| candidate OOD states | {a1['candidate_ood_count']} |",
            f"| candidate positive TVA on OOD | {a1['candidate_positive_tva_on_ood_count']} |",
            f"| selected OOD states | {a1['selected_ood_count']} |",
            f"| selected positive TVA on OOD | {a1['selected_positive_tva_on_ood_count']} |",
            f"| selected active positive TVA on OOD | {a1['selected_active_positive_tva_on_ood_count']} |",
            f"| selected z mean / max | {_fmt(a1['selected_feature_max_abs_z']['mean'])} / {_fmt(a1['selected_feature_max_abs_z']['max'])} |",
            "",
            "## A2：room-order 与承诺代理",
            "",
            "| 指标 | 数值 |",
            "|---|---:|",
            f"| frontier fill count | {a2['frontier_fill_count']} |",
            f"| non-frontier fill count | {a2['nonfrontier_fill_count']} |",
            f"| uncommitted entry transitions | {a2['uncommitted_entry_transition_count']} |",
            f"| uncommitted deep pushes | {a2['uncommitted_deep_push_count']} |",
            f"| room-order debt positive delta sum | {a2['room_order_debt_delta_sum']} |",
            f"| blocked-slot positive delta sum | {a2['blocked_slot_delta_sum']} |",
            f"| selected P-down/Q-up steps | {a2['p_down_q_up_selected_steps']} |",
            "",
            "## 前缀逐步摘要",
            "",
            "| step | selected | physical children | semantic proxy | impact codes | z | alpha | safe reason | room delta |",
            "|---:|---|---:|---:|---|---:|---:|---|---|",
        ]
    )
    for row in payload["trace_rows"]:
        cand = row["selected_candidate"]
        delta = cand["room_delta_totals"]
        delta_text = ",".join(
            f"{key}:{value}"
            for key, value in delta.items()
            if value and key in {"q_settled", "k_blocked", "uncommitted_entry", "uncommitted_in_room"}
        )
        lines.append(
            "| "
            f"{row['step']} | {row['selected_action_label']} | {row['physical_child_count_log']} | "
            f"{row['semantic_child_count_proxy_total']} | {','.join(cand['room_action']['impact_codes']) or 'none'} | "
            f"{_fmt(cand['child_feature_max_abs_z'])} | {_fmt(cand['child_current_alpha'])} | "
            f"{cand['child_safe_reason']} | {delta_text or '0'} |"
        )

    lines.extend(
        [
            "",
            "## 资源变化",
            "",
            "| resource | step 0 root | step 20 after | delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for key in ("P", "M", "A", "K", "D", "Q", "C"):
        before = trace0["root_resources"][key]
        after = tracelast["after_resources"][key]
        lines.append(f"| `{key}` | {_fmt(before)} | {_fmt(after)} | {_fmt(after - before)} |")
    lines.extend(
        [
            "",
            "## 方法含义",
            "",
            (
                "A1 是必要的安全门槛：在 XSokoban 这种明显越界状态上，TVA 不应允许分布外正向修正覆盖显式房间风险。"
                "A2 则表明，当前失败 trace 已经可以被解释为 room-slot frontier、无承诺 entry 和顺序债务问题。"
                "下一步 A3 应在新的 hard-instance 派生 planner 中把物理 child 扩展为 `(box,direction,slot)` 语义 child，"
                "并把这些承诺状态纳入节点 identity，而不是继续只在物理状态上合并。"
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
    parser.add_argument("--ood-z-threshold", type=float, default=3.0)
    parser.add_argument(
        "--out-json",
        default="results/sokoban_sd_world_model/room_gft_xsokoban_a1_a2_diagnostics.json",
    )
    parser.add_argument("--out-md", default="docs/room_gft_xsokoban_a1_a2_diagnostics.md")
    args = parser.parse_args()

    failure_path = ROOT / args.failure_log
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    config = failure["config"]
    framework = failure.get("framework_contract", {})
    level = load_boxoban_levels(ROOT / config["levels"], max_levels=int(config["level_offset"]) + 1)[
        int(config["level_offset"])
    ]
    model = build_room_model(level, config)
    tva_path = framework.get("tva_checkpoint")
    checkpoint = load_tva_checkpoint(ROOT / tva_path) if tva_path else None
    if checkpoint is not None and not isinstance(checkpoint, GFTTvaCheckpoint):
        checkpoint = None

    state = level.initial_state
    visited_keys = {model.canonical_key(state)}
    rows: list[dict[str, Any]] = []
    for trace in failure["levels"][0]["trace"]:
        step = int(trace["step"])
        selected = SokobanPushAction(
            box_index=int(trace["action"]["box_index"]),
            direction=int(trace["action"]["direction"]),
        )
        root_resources = resource_values(model, state)
        root_room = model.room_state_summary(state)
        candidates = candidate_diagnostics(
            model,
            state,
            step,
            selected,
            visited_keys,
            config,
            checkpoint,
            float(args.ood_z_threshold),
        )
        selected_rows = [row for row in candidates if row["selected"]]
        if not selected_rows:
            raise RuntimeError(f"Saved selected action is not legal at trace step {step}.")
        selected_row = selected_rows[0]
        selected_key = tuple(selected_row["action_key"])
        next_state = model.step_grid_flow(state, selected)
        after_resources = resource_values(model, next_state)
        after_room = model.room_state_summary(next_state)
        rows.append(
            {
                "step": step,
                "selected_action": selected_row["action"],
                "selected_action_label": action_label(selected),
                "physical_child_count_log": int(trace["child_count"]),
                "candidate_count_recomputed": len(candidates),
                "semantic_child_count_proxy_total": int(
                    sum(cand["room_semantic_options"]["semantic_child_count_proxy"] for cand in candidates)
                ),
                "selected_prior_rank": rank_of(candidates, selected_key, "prior_score", reverse=True),
                "selected_current_leaf_rank": rank_of(candidates, selected_key, "child_current_leaf_value", reverse=True),
                "selected_safe_leaf_rank": rank_of(candidates, selected_key, "child_safe_leaf_value", reverse=True),
                "root_resources": root_resources,
                "after_resources": after_resources,
                "root_room": root_room,
                "after_room": after_room,
                "room_delta_totals": _delta_totals(root_room, after_room),
                "selected_candidate": selected_row,
                "candidate_rows": sorted(
                    candidates,
                    key=lambda row: (row["child_safe_leaf_value"], row["prior_score"]),
                    reverse=True,
                ),
            }
        )
        state = next_state
        visited_keys.add(model.canonical_key(state))

    final_state = state
    payload = {
        "contract": {
            "analysis": "room_gft_xsokoban_a1_a2_diagnostics_v1",
            "failure_log": _rel(failure_path),
            "tva_checkpoint": tva_path,
            "level_id": level.level_id,
            "trace_scope": "saved_trace_prefix_only",
            "base_model": "SokobanGFTSDModel",
            "derived_model": "SokobanRoomGFTSDModel",
            "mainline_mutation": False,
        },
        "config_extract": {
            "horizon": int(config["horizon"]),
            "budget": int(config["budget"]),
            "max_steps": int(config["max_steps"]),
            "leaf_value": config["leaf_value"],
            "select_by": config["select_by"],
            "tva_state_gated_ood_z_threshold_current": config["tva_state_gated_ood_z_threshold"],
            "a1_ood_z_threshold": float(args.ood_z_threshold),
        },
        "room_static_summary": model.room_static_summary(),
        "a1_ood_safe_tva_summary": summarize_a1(rows, float(args.ood_z_threshold)),
        "a2_room_summary": summarize_a2(rows),
        "trace_rows": rows,
        "final_replayed_prefix_state": {
            "resources": resource_values(model, final_state),
            "room": model.room_state_summary(final_state),
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
