#!/usr/bin/env python3
"""Build stage-specific Hopper reset pools from saved official_eval traces."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel


STAGE_NAMES = ("landing", "ground", "rise", "prone_low", "prone_lift")
STATE_NAME_BY_ID = {
    0: "falling_air",
    1: "grounded_rear",
    2: "grounded_prone",
    3: "grounded_upright",
    4: "grounded_prepared",
    5: "rising_air",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 Hopper official_eval 轨迹中抽取阶段化 reset 池。"
    )
    parser.add_argument(
        "--trace-dir",
        action="append",
        default=None,
        help="包含 episode_XX.csv 的目录，可重复传入。",
    )
    parser.add_argument(
        "--trace-csv",
        action="append",
        default=None,
        help="直接指定单个 episode csv，可重复传入。",
    )
    parser.add_argument(
        "--stage",
        choices=STAGE_NAMES,
        required=True,
        help="要构造的阶段池：landing / ground / rise。",
    )
    parser.add_argument("--out-csv", type=Path, required=True, help="输出 reset 池 csv。")
    parser.add_argument("--out-json", type=Path, default=None, help="可选的汇总 json。")
    parser.add_argument("--max-per-trace", type=int, default=16, help="每条轨迹最多保留多少帧。")
    parser.add_argument("--max-total", type=int, default=2000, help="总样本上限。")
    parser.add_argument("--seed", type=int, default=0, help="随机种子，用于抽样。")
    args = parser.parse_args()
    if not args.trace_dir and not args.trace_csv:
        parser.error("至少需要传入 --trace-dir 或 --trace-csv")
    return args


def row_to_state(row: dict[str, str]) -> torch.Tensor:
    values = [
        float(row["torso_z"]),
        float(row["vx"]),
        float(row["vz"]),
        float(row["pitch"]),
        float(row["pitch_rate"]),
        float(row["activation"]),
        float(row["qpos_waist"]),
        float(row["qpos_hip"]),
        float(row["qpos_knee"]),
        float(row["qpos_ankle"]),
        float(row["qvel_waist"]),
        float(row["qvel_hip"]),
        float(row["qvel_knee"]),
        float(row["qvel_ankle"]),
        float(row["touch_toe"]),
        float(row["touch_heel"]),
        float(row["z_min"]),
    ]
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)


def load_rows(trace_dir: Path) -> list[tuple[Path, list[dict[str, str]]]]:
    paths = sorted(trace_dir.glob("episode_*.csv"))
    if not paths:
        raise FileNotFoundError(f"no episode csv found under {trace_dir}")
    loaded = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            loaded.append((path, list(csv.DictReader(f))))
    return loaded


def load_csv(path: Path) -> tuple[Path, list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        return path, list(csv.DictReader(f))


def is_stage_match(stage: str, state_id: int, feat: dict[str, torch.Tensor]) -> bool:
    if stage == "landing":
        return (
            state_id == 0
            and float(feat["airborne_true"][0].item()) > 0.5
            and float(feat["vz"][0].item()) < -0.10
        )
    if stage == "ground":
        return state_id in {1, 2, 3}
    if stage == "rise":
        if state_id == 4:
            return True
        return (
            state_id == 3
            and float(feat["upright_quality_gate"][0].item()) > 0.5
            and float(feat["full_foot_support_gate"][0].item()) > 0.5
            and float(feat["body_collision"][0].item()) < 0.5
        )
    if stage in {"prone_low", "prone_lift"}:
        if state_id != 2:
            return False
        torso_z = float(feat["torso_z"][0].item())
        support_contact = float(feat["support_contact"][0].item())
        nose_minus_anchor_x = float(feat["nose_minus_anchor_x"][0].item())
        if nose_minus_anchor_x < 0.10:
            return False
        if stage == "prone_low":
            return torso_z <= 0.45 and support_contact < 0.30
        return 0.40 < torso_z <= 0.75 and support_contact >= 0.30
    raise ValueError(f"unknown stage: {stage}")


def select_rows(
    model: DMControlHopperSDModel,
    stage: str,
    loaded_rows: list[tuple[Path, list[dict[str, str]]]],
    max_per_trace: int,
    max_total: int,
    seed: int,
) -> tuple[list[dict[str, str]], dict]:
    gen = torch.Generator().manual_seed(seed)
    selected: list[dict[str, str]] = []
    state_counter: Counter[str] = Counter()
    trace_counter: Counter[str] = Counter()

    for path, rows in loaded_rows:
        matches = []
        for row in rows:
            state = row_to_state(row)
            feat = model._state_reward_features(state)
            state_id = int(model._dominant_state_ids(feat)[0].item())
            if not is_stage_match(stage, state_id, feat):
                continue
            matches.append({
                "stage": stage,
                "state_id": state_id,
                "state_name": STATE_NAME_BY_ID[state_id],
                "source_csv": str(path),
                "step": row["step"],
                "torso_z": row["torso_z"],
                "vx": row["vx"],
                "vz": row["vz"],
                "pitch": row["pitch"],
                "pitch_rate": row["pitch_rate"],
                "touch_toe": row["touch_toe"],
                "touch_heel": row["touch_heel"],
                "z_min": row["z_min"],
                "qpos_rootx": row["qpos_rootx"],
                "qpos_rootz": row["qpos_rootz"],
                "qpos_rooty": row["qpos_rooty"],
                "qpos_waist": row["qpos_waist"],
                "qpos_hip": row["qpos_hip"],
                "qpos_knee": row["qpos_knee"],
                "qpos_ankle": row["qpos_ankle"],
                "qvel_rootx": row["qvel_rootx"],
                "qvel_rootz": row["qvel_rootz"],
                "qvel_rooty": row["qvel_rooty"],
                "qvel_waist": row["qvel_waist"],
                "qvel_hip": row["qvel_hip"],
                "qvel_knee": row["qvel_knee"],
                "qvel_ankle": row["qvel_ankle"],
            })
        if len(matches) > max_per_trace:
            order = torch.randperm(len(matches), generator=gen).tolist()
            matches = [matches[i] for i in order[:max_per_trace]]
        selected.extend(matches)
        trace_counter[str(path)] = len(matches)
        for row in matches:
            state_counter[row["state_name"]] += 1

    if len(selected) > max_total:
        order = torch.randperm(len(selected), generator=gen).tolist()
        selected = [selected[i] for i in order[:max_total]]
        state_counter = Counter(row["state_name"] for row in selected)

    summary = {
        "stage": stage,
        "num_selected": len(selected),
        "state_counts": dict(sorted(state_counter.items())),
        "trace_counts": dict(sorted(trace_counter.items())),
        "max_per_trace": max_per_trace,
        "max_total": max_total,
        "seed": seed,
    }
    return selected, summary


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("no reset samples selected")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    model = DMControlHopperSDModel().eval()
    loaded = []
    for trace_dir in args.trace_dir or []:
        loaded.extend(load_rows(Path(trace_dir)))
    for trace_csv in args.trace_csv or []:
        loaded.append(load_csv(Path(trace_csv)))
    selected, summary = select_rows(
        model=model,
        stage=args.stage,
        loaded_rows=loaded,
        max_per_trace=args.max_per_trace,
        max_total=args.max_total,
        seed=args.seed,
    )
    write_csv(args.out_csv, selected)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"out_csv={args.out_csv}")
    if args.out_json is not None:
        print(f"out_json={args.out_json}")


if __name__ == "__main__":
    main()
