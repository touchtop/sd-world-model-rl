#!/usr/bin/env python3
"""Compare Boxoban and XSokoban structural statistics for SD-MCTS analysis.

The script measures static geometry and shallow push-graph quantities for all
levels, then uses existing result logs to connect the structural shift to the
observed Boxoban success and XSokoban failure.  It deliberately avoids calling
the GFT matching DP on large-box XSokoban levels, because that DP is exact and
exponential in the number of boxes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, deque
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sokoban import SokobanGridStockFlowModel, SokobanLevel, SokobanState, load_boxoban_levels
from src.sokoban.env import DIRECTIONS, add_cells, render_state
from src.sokoban.gft_sd import SokobanGFTSDModel


def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def floor_neighbors(level: SokobanLevel, cell: tuple[int, int]) -> list[tuple[int, int]]:
    return [add_cells(cell, direction) for direction in DIRECTIONS if level.is_floor(add_cells(cell, direction))]


def articulation_points(level: SokobanLevel) -> set[tuple[int, int]]:
    """Return articulation points in the static floor graph."""

    floor = [(r, c) for r in range(level.height) for c in range(level.width) if level.is_floor((r, c))]
    if not floor:
        return set()
    visited: set[tuple[int, int]] = set()
    disc: dict[tuple[int, int], int] = {}
    low: dict[tuple[int, int], int] = {}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {}
    cuts: set[tuple[int, int]] = set()
    time = 0

    def dfs(u: tuple[int, int]) -> None:
        nonlocal time
        visited.add(u)
        disc[u] = low[u] = time
        time += 1
        child_count = 0
        for v in floor_neighbors(level, u):
            if v not in visited:
                parent[v] = u
                child_count += 1
                dfs(v)
                low[u] = min(low[u], low[v])
                if parent.get(u) is None and child_count > 1:
                    cuts.add(u)
                if parent.get(u) is not None and low[v] >= disc[u]:
                    cuts.add(u)
            elif v != parent.get(u):
                low[u] = min(low[u], disc[v])

    for cell in floor:
        if cell not in visited:
            parent[cell] = None
            dfs(cell)
    return cuts


def connected_components_without(level: SokobanLevel, blocked: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    floor = {
        (r, c)
        for r in range(level.height)
        for c in range(level.width)
        if level.is_floor((r, c)) and (r, c) not in blocked
    }
    components: list[set[tuple[int, int]]] = []
    while floor:
        start = next(iter(floor))
        comp = {start}
        queue: deque[tuple[int, int]] = deque([start])
        floor.remove(start)
        while queue:
            cell = queue.popleft()
            for nxt in floor_neighbors(level, cell):
                if nxt in floor:
                    floor.remove(nxt)
                    comp.add(nxt)
                    queue.append(nxt)
        components.append(comp)
    return components


def min_pairwise_manhattan(cells: Iterable[tuple[int, int]]) -> float | None:
    values = list(cells)
    if len(values) < 2:
        return None
    best = min(abs(a[0] - b[0]) + abs(a[1] - b[1]) for i, a in enumerate(values) for b in values[i + 1 :])
    return float(best)


def level_metrics(level: SokobanLevel) -> dict[str, Any]:
    model = SokobanGridStockFlowModel(level)
    state = level.initial_state
    floor_count = len(model.floor_cells)
    area = level.height * level.width
    reachable = model.reachable_cells(state)
    legal = model.legal_pushes(state)
    viable = 0
    for action in legal:
        if not model.is_tier_a_deadlocked(model.step_state(state, action)):
            viable += 1
    cuts = articulation_points(level)
    comps_without_cuts = connected_components_without(level, cuts)
    nontrivial_goal_components = sum(1 for comp in comps_without_cuts if comp & set(level.goals))
    dead_square_count = len(model.dead_squares)
    boxes_on_dead_squares = sum(1 for box in state.boxes if box in model.dead_squares and box not in level.goals)
    return {
        "level_id": level.level_id,
        "height": level.height,
        "width": level.width,
        "area": area,
        "floor_count": floor_count,
        "wall_count": area - floor_count,
        "wall_fraction": (area - floor_count) / area if area else None,
        "n_boxes": level.n_boxes,
        "n_goals": len(level.goals),
        "box_density_floor": level.n_boxes / floor_count if floor_count else None,
        "initial_reachable_count": len(reachable),
        "initial_reachable_fraction": len(reachable) / floor_count if floor_count else None,
        "initial_legal_push_count": len(legal),
        "initial_viable_push_count": viable,
        "legal_push_per_box": len(legal) / max(level.n_boxes, 1),
        "viable_push_per_box": viable / max(level.n_boxes, 1),
        "dead_square_count": dead_square_count,
        "dead_square_fraction_floor": dead_square_count / floor_count if floor_count else None,
        "initial_boxes_on_dead_squares": boxes_on_dead_squares,
        "articulation_count": len(cuts),
        "articulation_fraction_floor": len(cuts) / floor_count if floor_count else None,
        "components_without_articulations": len(comps_without_cuts),
        "nontrivial_goal_components_without_articulations": nontrivial_goal_components,
        "min_box_box_manhattan": min_pairwise_manhattan(state.boxes),
        "min_goal_goal_manhattan": min_pairwise_manhattan(level.goals),
    }


def summarize_numeric(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, dict[str, float | int | None]]:
    out: dict[str, dict[str, float | int | None]] = {}
    for key in keys:
        vals = [row[key] for row in rows if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))]
        vals = [float(v) for v in vals]
        if not vals:
            out[key] = {"mean": None, "median": None, "min": None, "max": None}
            continue
        vals_sorted = sorted(vals)
        out[key] = {
            "mean": mean(vals),
            "median": median(vals),
            "min": vals_sorted[0],
            "p90": vals_sorted[int(0.9 * (len(vals_sorted) - 1))],
            "max": vals_sorted[-1],
        }
    return out


def boxoban_status_by_level(summary_path: Path) -> dict[int, str]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    status: dict[int, str] = {}
    for shard in summary["shards"]:
        shard_path = ROOT / shard["path"]
        payload = json.loads(shard_path.read_text(encoding="utf-8"))
        offset = int(payload["config"]["level_offset"])
        for row in payload["levels"]:
            status[offset + int(row["level_index"])] = str(row["status"])
    return status


def full_gft_initial(level: SokobanLevel) -> dict[str, Any]:
    model = SokobanGFTSDModel(
        level,
        transport_cost_weights={"turn": 0.1, "access": 0.25, "zone": 0.25},
        soft_deadlock_weights={
            "mobility_loss": 1.0,
            "matching_fragility": 1.0,
            "capacity_stress": 1.0,
            "access_loss": 1.0,
            "order_violation": 1.0,
        },
        commitment_beta=1.0,
        sinkhorn_iterations=25,
    )
    state = level.initial_state
    readout = model.gft_state(state, active_only=True).to_dict(active_only=True)
    return {
        "abstract_features": model.abstract_features(state).to_dict(),
        "tier_a_status": model.tier_a_status(state),
        "gft_resource_values": {key: value["value"] for key, value in readout["resources"].items()},
        "gft_readout": readout,
        "rendered_initial_state": render_state(level, state),
    }


def write_markdown(
    out_path: Path,
    *,
    box_summary: dict[str, Any],
    xs_summary: dict[str, Any],
    xs_first: dict[str, Any],
    xs_failure: dict[str, Any],
    box_fail_summary: dict[str, Any],
) -> None:
    lines = [
        "# XSokoban first 与 Boxoban unfiltered/test 结构差异分析",
        "",
        "## 结论摘要",
        "",
        (
            "Boxoban `unfiltered/test/000.txt` 的 1000 关是固定 10x10、4 箱、目标较分散的同分布测试；"
            "XSokoban first-40 则来自经典人工关卡，尺寸、箱数、目标密度和瓶颈结构均明显外推。"
        ),
        (
            "当前最佳 SD-MCTS 在 Boxoban 上达到 989/1000，主要依赖 4 箱下可精确计算且与训练/审计分布一致的"
            " GFT 运输匹配、局部 push-level 搜索和小规模 TVA 叶值校正。XSokoban first 第 1 关已经是 11x19、6 箱、"
            "目标集中成房间，`max_steps=1000` 仍未解出；这不是单纯步数上限问题，而是结构分布、目标分配复杂度、"
            "局部循环和 TVA/GFT 标定范围同时越界。"
        ),
        "",
        "## 数据与结果口径",
        "",
        f"- Boxoban 测试集：`{box_summary['path']}`，`{box_summary['count']}` 关。",
        f"- XSokoban first-40：`{xs_summary['path']}`，`{xs_summary['count']}` 关。",
        "- Boxoban 最佳结果：989/1000，失败类型为 7 个 `max_steps`、4 个 `deadlocked`。",
        (
            "- XSokoban 第 1 关结果：`max_steps=1000`，未解出，最终 `boxes_on_goals=1`，"
            f"耗时 {xs_failure['elapsed_sec']:.1f} 秒。"
        ),
        "",
        "## 结构统计对照",
        "",
        "| 指标 | Boxoban unfiltered/test | XSokoban first-40 | XSokoban 第 1 关 |",
        "|---|---:|---:|---:|",
    ]
    metric_labels = [
        ("height", "高度"),
        ("width", "宽度"),
        ("area", "网格面积"),
        ("floor_count", "可行走地板数"),
        ("n_boxes", "箱子数"),
        ("box_density_floor", "箱子/地板密度"),
        ("initial_reachable_count", "初始玩家可达格数"),
        ("initial_reachable_fraction", "初始玩家可达比例"),
        ("initial_legal_push_count", "初始合法 push 数"),
        ("legal_push_per_box", "初始合法 push/箱"),
        ("dead_square_fraction_floor", "dead-square 地板占比"),
        ("articulation_count", "静态割点数"),
        ("components_without_articulations", "移除割点后的组件数"),
        ("min_goal_goal_manhattan", "最近目标曼哈顿距"),
    ]
    for key, label in metric_labels:
        box_val = box_summary["numeric_summary"][key]["median"]
        xs_val = xs_summary["numeric_summary"][key]["median"]
        first_val = xs_first[key]
        lines.append(f"| {label}（中位数） | {fmt(box_val)} | {fmt(xs_val)} | {fmt(first_val)} |")
    lines.extend(
        [
            "",
            "XSokoban first-40 的中位箱数为 "
            f"{fmt(xs_summary['numeric_summary']['n_boxes']['median'])}，最大箱数为 "
            f"{fmt(xs_summary['numeric_summary']['n_boxes']['max'])}；Boxoban 测试集箱数恒为 4。"
            "这直接突破了当前模型中若干按 4 箱分布调校的归纳偏置。XSokoban 第 1 关虽然只是 first-40 中较小的一关，"
            f"也已有 `{xs_first['n_boxes']}` 个箱子、`{xs_first['floor_count']}` 个地板格和 "
            f"`{xs_first['articulation_count']}` 个静态割点。",
            "",
            "## 与 Boxoban 失败关卡的关系",
            "",
            (
                "Boxoban 1000 关中仅 11 关失败。失败关卡的结构统计仍大体处在 10x10、4 箱分布内，"
                "与 XSokoban 第 1 关相比，主要差异不是单个指标略难，而是 XSokoban 同时改变了尺寸、箱数、目标组织方式和走廊/房间结构。"
            ),
            "",
            "| 指标 | Boxoban 全体中位数 | Boxoban 失败关卡中位数 | XSokoban 第 1 关 |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, label in metric_labels:
        lines.append(
            f"| {label} | {fmt(box_summary['numeric_summary'][key]['median'])} | "
            f"{fmt(box_fail_summary['numeric_summary'][key]['median'])} | {fmt(xs_first[key])} |"
        )
    lines.extend(
        [
            "",
            "## 失败机制解释",
            "",
            "1. **组合规模外推**：Boxoban 测试集固定 4 箱，XSokoban first-40 至少 6 箱，部分关卡达到数十箱。"
            "当前 GFT 的精确箱-目标匹配、Sinkhorn commitment 和 ordering/capacity 读数在 4 箱上稳定，"
            "但在多箱目标房间中会变成高熵、多等价分配问题。XSokoban 第 1 关初始 commitment 仅为 "
            f"`{fmt(xs_failure['initial_gft']['gft_resource_values']['C'])}`，说明箱-目标绑定本身很弱。",
            "",
            "2. **目标集中与房间顺序约束增强**：Boxoban unfiltered 常见目标较分散，匹配下界下降即可提供有效局部进展信号。"
            "XSokoban 第 1 关有 6 个目标集中在右下目标房间，进入顺序和填充顺序比单纯距离更关键；"
            "失败日志最终仍有 ordering debt 和 soft-deadlock pressure，说明模型能读出风险，但该读数没有转化为足够强的长程承诺。",
            "",
            "3. **push-level 局部搜索陷入低分支循环**：XSokoban `max_steps=1000` 日志中，前 20 步 trace 已出现多次"
            "同一箱子重复向右推动，许多 root 只有 1 个有效 child。提高 `max_steps` 允许更长执行，但不能改变"
            "每一步局部 MCTS 的动作排序和承诺机制，因此仍会在长走廊/目标房间外围消耗搜索。",
            "",
            "4. **TVA 训练与审计分布不覆盖 XSokoban**：当前 TVA value 训练来自 40 个 Boxoban/medium 切片状态，"
            "独立审计也是 Boxoban 叶状态；它没有见过 11x19、6 箱及更多箱、目标房间式 XSokoban 状态。"
            "因此 Boxoban 上有用的 `V_SD + TVA` 校正，在 XSokoban 上缺少可靠校准证据。",
            "",
            "5. **评价尺度不只是不够长**：第 1 关在 `max_steps=1000` 下仍只完成 1 个目标，最终 matching/transport 工作量仍高。"
            "这说明失败不只是 `80` 步上限偏短，而是规划前缀没有进入正确的目标填充策略。",
            "",
            "## 后续实验建议",
            "",
            "- 对 XSokoban 第 1 关先做离线可解性与最短/可行 push 参考解验证，作为诊断轨迹锚点。",
            "- 在不改变论文主结果的前提下，增加 `--trace-gft-readouts` 复跑 XSokoban 第 1 关前 50-100 步，检查 P/M/A/K/D/Q/C 的时间序列是否饱和或误导。",
            "- 为多箱目标房间加入显式 box-goal/room-order commitment option，而不是继续只放大 `max_steps`。",
            "- 若要系统评估泛化，应分开报告 Boxoban hard、XSokoban first-40 与大尺寸 XSokoban 子集；三者不能互相替代。",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def fmt(value: Any) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boxoban", default="datasets/sokoban_boxoban/boxoban-levels/unfiltered/test/000.txt")
    parser.add_argument("--xsokoban", default="datasets/sokoban_xsokoban/xsokoban_first40_boxoban_format.txt")
    parser.add_argument(
        "--boxoban-summary",
        default="results/sokoban_sd_world_model/gft_sdmcts_boxoban_unfiltered_test1000_online_tva_state_gated_phase060_prefix5_rooteligible_verified_gates_seed0_sharded_summary.json",
    )
    parser.add_argument(
        "--xsokoban-result",
        default="results/sokoban_sd_world_model/xsokoban_first40_bestconfig_exact_l001_maxsteps1000_seed0.json",
    )
    parser.add_argument("--out-json", default="results/sokoban_sd_world_model/xsokoban_boxoban_structure_shift_analysis.json")
    parser.add_argument("--out-md", default="docs/sokoban_xsokoban_structure_shift_analysis.md")
    args = parser.parse_args()

    box_path = ROOT / args.boxoban
    xs_path = ROOT / args.xsokoban
    box_levels = load_boxoban_levels(box_path)
    xs_levels = load_boxoban_levels(xs_path)

    box_metrics = [level_metrics(level) for level in box_levels]
    xs_metrics = [level_metrics(level) for level in xs_levels]
    status_by_level = boxoban_status_by_level(ROOT / args.boxoban_summary)
    for idx, row in enumerate(box_metrics):
        row["status"] = status_by_level.get(idx, "unknown")
    box_fail_metrics = [row for row in box_metrics if row.get("status") != "solved"]

    numeric_keys = [
        "height",
        "width",
        "area",
        "floor_count",
        "wall_count",
        "wall_fraction",
        "n_boxes",
        "box_density_floor",
        "initial_reachable_count",
        "initial_reachable_fraction",
        "initial_legal_push_count",
        "initial_viable_push_count",
        "legal_push_per_box",
        "viable_push_per_box",
        "dead_square_count",
        "dead_square_fraction_floor",
        "initial_boxes_on_dead_squares",
        "articulation_count",
        "articulation_fraction_floor",
        "components_without_articulations",
        "nontrivial_goal_components_without_articulations",
        "min_box_box_manhattan",
        "min_goal_goal_manhattan",
    ]

    xs_result = json.loads((ROOT / args.xsokoban_result).read_text(encoding="utf-8"))
    xs_level_result = xs_result["levels"][0]
    xs_initial_gft = full_gft_initial(xs_levels[0])
    xs_failure = {
        "result_path": _rel(ROOT / args.xsokoban_result),
        "status": xs_level_result["status"],
        "steps_executed": xs_level_result["steps_executed"],
        "elapsed_sec": xs_level_result["elapsed_sec"],
        "final_features": xs_level_result["final_features"],
        "final_gft_resource_values": {
            key: value["value"] for key, value in xs_level_result["final_gft_state"]["resources"].items()
        },
        "initial_gft": xs_initial_gft,
        "trace_prefix_actions": [
            {
                "step": row["step"],
                "action": row["action"]["direction_name"],
                "box_index": row["action"]["box_index"],
                "child_count": row["child_count"],
                "boxes_on_goals_after": row["boxes_on_goals_after"],
            }
            for row in xs_level_result["trace"][:20]
        ],
    }

    payload = {
        "contract": {
            "analysis": "xsokoban_first_vs_boxoban_unfiltered_test_structure_shift_v1",
            "boxoban_path": _rel(box_path),
            "xsokoban_path": _rel(xs_path),
            "boxoban_result_path": _rel(ROOT / args.boxoban_summary),
            "xsokoban_result_path": _rel(ROOT / args.xsokoban_result),
        },
        "boxoban": {
            "path": _rel(box_path),
            "count": len(box_levels),
            "status_counts": dict(Counter(row["status"] for row in box_metrics)),
            "numeric_summary": summarize_numeric(box_metrics, numeric_keys),
        },
        "boxoban_failed": {
            "count": len(box_fail_metrics),
            "level_indices": [idx for idx, row in enumerate(box_metrics) if row.get("status") != "solved"],
            "status_counts": dict(Counter(row["status"] for row in box_fail_metrics)),
            "numeric_summary": summarize_numeric(box_fail_metrics, numeric_keys),
            "rows": box_fail_metrics,
        },
        "xsokoban_first40": {
            "path": _rel(xs_path),
            "count": len(xs_levels),
            "box_count_distribution": dict(Counter(row["n_boxes"] for row in xs_metrics)),
            "numeric_summary": summarize_numeric(xs_metrics, numeric_keys),
            "rows": xs_metrics,
        },
        "xsokoban_first_level": {
            "metrics": xs_metrics[0],
            "failure": xs_failure,
        },
    }

    out_json = ROOT / args.out_json
    out_md = ROOT / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(
        out_md,
        box_summary=payload["boxoban"],
        xs_summary=payload["xsokoban_first40"],
        xs_first=payload["xsokoban_first_level"]["metrics"],
        xs_failure=payload["xsokoban_first_level"]["failure"],
        box_fail_summary=payload["boxoban_failed"],
    )
    print(json.dumps({"out_json": _rel(out_json), "out_md": _rel(out_md)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
