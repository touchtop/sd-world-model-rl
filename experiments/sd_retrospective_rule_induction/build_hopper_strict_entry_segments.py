#!/usr/bin/env python3
"""Build stricter Hopper ability-entry segment datasets.

The original ability-entry dataset was designed for tracing how an ability
enters replay, so it includes pre/post cycles and allows the same window to
appear under multiple thresholds.  This script derives stricter datasets for
rule-learning audits without modifying the original files.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import write_json, load_json, mean_or_zero  # noqa: E402


DEFAULT_SEGMENTS = Path(
    "results/sd_retrospective_rule_induction/ability_entry/"
    "hopper_teacher_core_seed123_20260522_r1/entry_candidate_segments.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "results/sd_retrospective_rule_induction/ability_entry/"
    "hopper_teacher_core_seed123_20260522_r1"
)


def window_uid(segment: dict[str, Any], window: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(segment.get("trace_csv", "")),
        int(window.get("anchor_index", -1)),
        str(window.get("anchor_type", "")),
    )


def flatten(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seg_idx, segment in enumerate(payload.get("segments", [])):
        base = {k: v for k, v in segment.items() if k != "windows"}
        for win_idx, window in enumerate(segment.get("windows", [])):
            rows.append(
                {
                    "segment_index": seg_idx,
                    "window_index": win_idx,
                    "base": base,
                    "window": dict(window),
                    "uid": window_uid(segment, window),
                    "role": str(segment["role"]),
                    "threshold": float(segment["threshold"]),
                    "cycle_offset": int(segment["cycle"]) - int(segment["anchor_cycle"]),
                    "future_reward_sum": float(window.get("future_reward_sum", 0.0) or 0.0),
                    "future_rootx_delta": float(window.get("future_rootx_delta", 0.0) or 0.0),
                }
            )
    return rows


def filter_by_mode(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode == "raw":
        return list(rows)
    if mode == "no_pre_entry":
        return [row for row in rows if row["cycle_offset"] >= 0]
    if mode == "entry_cycle_only":
        return [row for row in rows if row["cycle_offset"] == 0]
    raise ValueError("unknown mode: " + mode)


def remove_role_conflicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_uid: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_uid[row["uid"]].append(row)
    allowed = {
        uid for uid, vals in by_uid.items()
        if len({row["role"] for row in vals}) == 1
    }
    return [row for row in rows if row["uid"] in allowed]


def dedupe_by_uid(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_uid: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_uid[row["uid"]].append(row)
    selected = []
    for vals in by_uid.values():
        vals = sorted(
            vals,
            key=lambda row: (
                row["threshold"],
                -abs(row["cycle_offset"]),
                row["future_reward_sum"],
                row["future_rootx_delta"],
            ),
            reverse=True,
        )
        selected.append(vals[0])
    return sorted(
        selected,
        key=lambda row: (
            int(row["base"]["anchor_cycle"]),
            int(row["base"]["seed"]),
            int(row["base"]["cycle"]),
            int(row["base"]["episode"]),
            int(row["window"].get("anchor_index", -1)),
            row["role"],
            row["threshold"],
        ),
    )


def rows_to_segments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments = []
    for row in rows:
        segment = dict(row["base"])
        window = dict(row["window"])
        window["strict_source_uid"] = "|".join(str(x) for x in row["uid"])
        window["strict_original_threshold"] = row["threshold"]
        window["strict_cycle_offset"] = row["cycle_offset"]
        segment["windows"] = [window]
        segment["strict_label_mode"] = True
        segments.append(segment)
    return segments


def summarize(rows_before: list[dict[str, Any]], rows_after: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    role_counts = Counter(row["role"] for row in rows_after)
    threshold_counts = Counter(str(row["threshold"]) for row in rows_after)
    offset_counts = Counter(str(row["cycle_offset"]) for row in rows_after)
    by_uid_before: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows_before:
        by_uid_before[row["uid"]].append(row)
    role_conflicts_before = sum(1 for vals in by_uid_before.values() if len({row["role"] for row in vals}) > 1)
    return {
        "mode": mode,
        "raw_windows": len(rows_before),
        "selected_windows": len(rows_after),
        "unique_uids_selected": len({row["uid"] for row in rows_after}),
        "role_counts": dict(role_counts.most_common()),
        "threshold_counts": dict(sorted(threshold_counts.items())),
        "cycle_offset_counts": dict(sorted(offset_counts.items())),
        "role_conflict_uid_groups_before_filter": role_conflicts_before,
        "mean_future_reward_sum": mean_or_zero(row["future_reward_sum"] for row in rows_after),
        "mean_future_rootx_delta": mean_or_zero(row["future_rootx_delta"] for row in rows_after),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper Strict Entry Segments",
        "",
        f"Source: `{payload['source']}`",
        "",
        "| Mode | Raw | Selected | Unique | Roles | Offsets | Thresholds |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for item in payload["summaries"]:
        lines.append(
            f"| `{item['mode']}` | {item['raw_windows']} | {item['selected_windows']} | "
            f"{item['unique_uids_selected']} | `{item['role_counts']}` | "
            f"`{item['cycle_offset_counts']}` | `{item['threshold_counts']}` |"
        )
    lines.extend([
        "",
        "## Label Policy",
        "",
        "严格数据集移除同一窗口在不同事件中产生 success/failure 角色冲突的样本，并对同一窗口跨阈值重复出现的情况只保留一个最高阈值实例。`entry_cycle_only` 仅保留入口当周期；`no_pre_entry` 保留入口当周期和入口后一周期。",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_mode(rows: list[dict[str, Any]], mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    filtered = filter_by_mode(rows, mode)
    filtered = remove_role_conflicts(filtered)
    filtered = dedupe_by_uid(filtered)
    return rows_to_segments(filtered), summarize(rows, filtered, mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--modes", nargs="+", default=["no_pre_entry", "entry_cycle_only"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_json(args.segments)
    rows = flatten(payload)
    summaries = []
    outputs = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        segments, summary = build_mode(rows, mode)
        out = args.output_dir / f"entry_candidate_segments_strict_{mode}.json"
        write_json(out, {"segments": segments, "strict_summary": summary, "source": str(args.segments)})
        summaries.append(summary)
        outputs.append(str(out))
    report = {
        "analysis_family": "hopper_strict_entry_segments_v1",
        "source": str(args.segments),
        "outputs": outputs,
        "summaries": summaries,
    }
    write_json(args.output_dir / "entry_candidate_segments_strict_summary.json", report)
    write_markdown(args.output_dir / "entry_candidate_segments_strict_summary.md", report)
    print(f"wrote strict entry segments: {args.output_dir}")
    for summary in summaries:
        print(summary["mode"], "selected", summary["selected_windows"], "roles", summary["role_counts"])


if __name__ == "__main__":
    main()
