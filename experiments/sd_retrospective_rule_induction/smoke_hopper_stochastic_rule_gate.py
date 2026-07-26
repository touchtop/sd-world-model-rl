"""Smoke test confidence-calibrated stochastic rule gating on saved Hopper traces.

The test is read-only. It reconstructs raw Hopper observations from official
eval CSV files and calls ``HopperRulePriorV6.score`` with synthetic candidate
actions to verify matched/applied/dropped rule metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from online_rule_prior_v6 import HopperRulePriorV6


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", "nan", "None"} else 0.0


def _obs_from_row(row: dict[str, str]) -> torch.Tensor:
    # dm_control Hopper observations are position[6], velocity[7], touch[2].
    values = [
        _float(row, "qpos_rootz"),
        _float(row, "qpos_rooty"),
        _float(row, "qpos_waist"),
        _float(row, "qpos_hip"),
        _float(row, "qpos_knee"),
        _float(row, "qpos_ankle"),
        _float(row, "qvel_rootx"),
        _float(row, "qvel_rootz"),
        _float(row, "qvel_rooty"),
        _float(row, "qvel_waist"),
        _float(row, "qvel_hip"),
        _float(row, "qvel_knee"),
        _float(row, "qvel_ankle"),
        _float(row, "touch_toe"),
        _float(row, "touch_heel"),
    ]
    return torch.tensor(values, dtype=torch.float32)


def _iter_rows(trace_dirs: list[Path], stride: int, max_rows: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for trace_dir in trace_dirs:
        for csv_path in sorted(trace_dir.glob("episode_*.csv")):
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    if idx % stride != 0:
                        continue
                    rows.append(row)
                    if len(rows) >= max_rows:
                        return rows
    return rows


def _make_actions(row: dict[str, str], horizon: int, num_samples: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    actions = torch.empty(horizon, num_samples, 4, dtype=torch.float32).uniform_(-1.0, 1.0, generator=generator)
    trace_action = torch.tensor(
        [_float(row, "action_0"), _float(row, "action_1"), _float(row, "action_2"), _float(row, "action_3")],
        dtype=torch.float32,
    ).clamp(-1.0, 1.0)
    actions[0, 0] = trace_action
    return actions


def _run_mode(
    rows: list[dict[str, str]],
    library: Path,
    *,
    stochastic_gate: bool,
    stochastic_eval: bool,
    eval_mode: bool,
    min_apply_prob: float,
    max_apply_prob: float,
    confidence_prob_power: float,
    horizon: int,
    num_samples: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    prior = HopperRulePriorV6(
        library_path=library,
        device=torch.device("cpu"),
        stage_mode="full",
        weight=0.05,
        opposite_penalty=0.01,
        max_bonus=0.10,
        mode="penalty_only",
        preferred_cap=0.70,
        cap_penalty=0.01,
        allowed_statuses={"confidence_active"},
        stochastic_gate=stochastic_gate,
        min_apply_prob=min_apply_prob,
        max_apply_prob=max_apply_prob,
        confidence_prob_power=confidence_prob_power,
        stochastic_eval=stochastic_eval,
    )
    matched = 0.0
    applied = 0.0
    dropped = 0.0
    prob_sum = 0.0
    multiplier_sum = 0.0
    hit_steps = 0
    bonus_abs_sum = 0.0
    max_bonus = 0.0
    for idx, row in enumerate(rows):
        obs = _obs_from_row(row)
        actions = _make_actions(row, horizon, num_samples, seed + idx)
        bonus = prior.score(obs, actions, eval_mode=eval_mode)
        metrics = prior.last_metrics
        step_matched = float(metrics.get("rule_prior_matched_rules", 0.0))
        if step_matched > 0:
            hit_steps += 1
            matched += step_matched
            applied += float(metrics.get("rule_prior_applied_rules", 0.0))
            dropped += float(metrics.get("rule_prior_dropped_rules", 0.0))
            prob_sum += float(metrics.get("rule_prior_apply_prob_mean", 0.0))
            multiplier_sum += float(metrics.get("rule_prior_apply_multiplier_mean", 0.0))
        bonus_abs_sum += float(bonus.abs().mean().item())
        max_bonus = max(max_bonus, float(bonus.abs().max().item()))
    return {
        "rows": len(rows),
        "hit_steps": hit_steps,
        "matched_rules": matched,
        "applied_rules": applied,
        "dropped_rules": dropped,
        "drop_rate": dropped / matched if matched else 0.0,
        "apply_rate": applied / matched if matched else 0.0,
        "mean_apply_prob_on_hit_steps": prob_sum / hit_steps if hit_steps else 0.0,
        "mean_apply_multiplier_on_hit_steps": multiplier_sum / hit_steps if hit_steps else 0.0,
        "mean_abs_bonus": bonus_abs_sum / len(rows) if rows else 0.0,
        "max_abs_bonus": max_bonus,
        "stochastic_gate": stochastic_gate,
        "stochastic_eval": stochastic_eval,
        "eval_mode": eval_mode,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = ["# Hopper Stochastic Rule Gate Smoke", ""]
    lines.append("## Config")
    lines.append("")
    for key, value in report["config"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| mode | rows | hit steps | matched | applied | dropped | apply rate | drop rate | mean prob | mean multiplier | mean abs bonus |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, row in report["results"].items():
        lines.append(
            f"| `{name}` | {row['rows']} | {row['hit_steps']} | {row['matched_rules']:.0f} | "
            f"{row['applied_rules']:.0f} | {row['dropped_rules']:.0f} | {row['apply_rate']:.3f} | "
            f"{row['drop_rate']:.3f} | {row['mean_apply_prob_on_hit_steps']:.3f} | "
            f"{row['mean_apply_multiplier_on_hit_steps']:.3f} | {row['mean_abs_bonus']:.5f} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "若 `train_stochastic` 的 apply rate 接近 mean prob，说明训练期随机门控按信心概率工作；"
        "`eval_expected` 的 dropped 应为 0，mean multiplier 应接近 mean prob，说明 eval 使用确定性期望缩放。"
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=1500)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-apply-prob", type=float, default=0.10)
    parser.add_argument("--max-apply-prob", type=float, default=0.95)
    parser.add_argument("--confidence-prob-power", type=float, default=1.0)
    args = parser.parse_args()

    rows = _iter_rows(args.trace_dir, args.stride, args.max_rows)
    if not rows:
        raise RuntimeError("No trace rows found for smoke test.")

    common = dict(
        rows=rows,
        library=args.library,
        min_apply_prob=args.min_apply_prob,
        max_apply_prob=args.max_apply_prob,
        confidence_prob_power=args.confidence_prob_power,
        horizon=args.horizon,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    report = {
        "config": {
            "library": str(args.library),
            "trace_dirs": [str(p) for p in args.trace_dir],
            "stride": args.stride,
            "max_rows": args.max_rows,
            "actual_rows": len(rows),
            "horizon": args.horizon,
            "num_samples": args.num_samples,
            "seed": args.seed,
            "min_apply_prob": args.min_apply_prob,
            "max_apply_prob": args.max_apply_prob,
            "confidence_prob_power": args.confidence_prob_power,
        },
        "results": {
            "deterministic_off": _run_mode(
                **common,
                stochastic_gate=False,
                stochastic_eval=False,
                eval_mode=False,
            ),
            "train_stochastic": _run_mode(
                **common,
                stochastic_gate=True,
                stochastic_eval=False,
                eval_mode=False,
            ),
            "eval_expected": _run_mode(
                **common,
                stochastic_gate=True,
                stochastic_eval=False,
                eval_mode=True,
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(args.output.with_suffix(".md"), report)
    print(f"wrote {args.output}")
    print(f"markdown {args.output.with_suffix('.md')}")
    for name, row in report["results"].items():
        print(
            f"{name}: hit_steps={row['hit_steps']} matched={row['matched_rules']:.0f} "
            f"applied={row['applied_rules']:.0f} dropped={row['dropped_rules']:.0f} "
            f"apply_rate={row['apply_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
