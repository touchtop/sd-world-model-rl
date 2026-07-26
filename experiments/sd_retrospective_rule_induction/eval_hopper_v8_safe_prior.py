#!/usr/bin/env python3
"""Eval-only comparison for v8 replay-safe transition prior candidates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = REPO_ROOT / "experiments" / "sd_retrospective_rule_induction" / "eval_hopper_v7_rule_prior.py"
PYTHON = "/home/anaconda3/envs/sdmcts_p100/bin/python"
RUN_NAME = "tdmpc2_v72_multiseed_shared_5k_seed123_20260507_r3_cycle040_target200039"
CLOSED_LOOP_ROOT = (
    REPO_ROOT
    / "results"
    / "sd_retrospective_rule_induction"
    / "closed_loop"
    / "hopper_v72_multiseed_shared_200k_seed123_resume100k_20260507_r2"
)
V8_SAFE_RULES = (
    REPO_ROOT
    / "results"
    / "sd_retrospective_rule_induction"
    / "rules"
    / "hopper_v8_transition_replay_safe_rules_200k_v72.json"
)


def checkpoint(seed: int) -> Path:
    return REPO_ROOT / "logs" / "hopper-hop" / str(seed) / RUN_NAME / "models" / "final.full.pt"


def v7_rules() -> Path:
    return CLOSED_LOOP_ROOT / "cycle_040" / "rules" / "library_closed_loop_cycle_040.json"


def parse_reward_list(text: str) -> list[float]:
    return [float(x) for x in str(text).split(";") if x != ""]


def mode_args(mode: str) -> list[str]:
    if mode == "rule_off":
        return ["+rule_prior_enabled=false"]
    if mode == "v7_current":
        return [
            "+rule_prior_enabled=true",
            f"+rule_prior_library={v7_rules()}",
            "+rule_prior_allowed_statuses='replay_validated,rolling_validated'",
            "+rule_prior_mode=penalty_only",
            "+rule_prior_opposite_penalty=0.02",
            "+rule_prior_preferred_cap=0.70",
            "+rule_prior_cap_penalty=0.02",
        ]
    if mode == "v8_safe":
        return [
            "+rule_prior_enabled=true",
            f"+rule_prior_library={V8_SAFE_RULES}",
            "+rule_prior_allowed_statuses=v8_replay_safe_candidate",
            "+rule_prior_mode=penalty_only",
            "+rule_prior_opposite_penalty=0.02",
            "+rule_prior_preferred_cap=0.70",
            "+rule_prior_cap_penalty=0.02",
            "+rule_prior_v8_continue_score_min=0.60",
        ]
    raise ValueError(f"unknown mode: {mode}")


def run_eval(
    *,
    seed: int,
    gpu_id: int,
    mode: str,
    eval_episodes: int,
    output_dir: Path,
    dry_run: bool,
    resume: bool,
) -> Path:
    ckpt = checkpoint(seed)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    if mode == "v7_current" and not v7_rules().exists():
        raise FileNotFoundError(v7_rules())
    if mode == "v8_safe" and not V8_SAFE_RULES.exists():
        raise FileNotFoundError(V8_SAFE_RULES)

    report = output_dir / f"seed{seed}_{mode}.json"
    if resume and report.exists():
        print(f"skip existing report: {report}", flush=True)
        return report
    exp_name = f"tdmpc2_v8_safe_eval_seed{seed}_{mode}_20260508"
    cmd = [
        PYTHON,
        str(EVAL_SCRIPT),
        "task=hopper-hop",
        "world_model_kind=tdmpc2",
        "model_size=null",
        "obs=state",
        f"seed={seed}",
        f"exp_name={exp_name}",
        "enable_wandb=false",
        "save_video=false",
        "save_agent=false",
        "save_csv=true",
        "save_full_state=false",
        "save_eval_traces=true",
        f"eval_episodes={eval_episodes}",
        "compile=false",
        "batch_size=128",
        "lr=3e-4",
        "dropout=0.01",
        "mlp_dim=256",
        "num_q=3",
        "num_samples=64",
        "num_elites=8",
        "num_pi_trajs=8",
        "horizon=3",
        "iterations=3",
        "sd_stage_mode=full",
        f"+checkpoint={ckpt}",
        "+eval_action_mode=eval",
        f"+eval_report={report}",
        *mode_args(mode),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["MUJOCO_GL"] = "glfw"
    env["PYTHONPATH"] = f"{REPO_ROOT / 'external' / 'tdmpc2' / 'tdmpc2'}:{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    return report


def summarize(output_dir: Path, seeds: list[int], modes: list[str], summary: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for mode in modes:
            path = output_dir / f"seed{seed}_{mode}.json"
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            rewards = parse_reward_list(payload["episode_reward_list"])
            rows.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "mean": float(payload["episode_reward"]),
                    "std": pstdev(rewards) if len(rewards) > 1 else 0.0,
                    "min": min(rewards) if rewards else 0.0,
                    "max": max(rewards) if rewards else 0.0,
                    "rewards": rewards,
                    "rule_prior_num_rules": payload.get("rule_prior_num_rules", 0),
                    "report": str(path),
                }
            )

    by_mode: dict[str, dict[str, Any]] = {}
    for mode in modes:
        vals = [row["mean"] for row in rows if row["mode"] == mode]
        by_mode[mode] = {
            "seed_mean_values": vals,
            "mean_across_seeds": mean(vals) if vals else 0.0,
            "std_across_seeds": pstdev(vals) if len(vals) > 1 else 0.0,
        }
    payload = {"rows": rows, "by_mode": by_mode}
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    write_markdown(summary.with_suffix(".md"), payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hopper v8 Replay-safe Prior Eval-only Sweep",
        "",
        "| mode | seed means | mean across seeds | std across seeds |",
        "| --- | --- | ---: | ---: |",
    ]
    for mode, stats in payload["by_mode"].items():
        seed_means = "; ".join(f"{x:.3f}" for x in stats["seed_mean_values"])
        lines.append(
            f"| `{mode}` | `{seed_means}` | {stats['mean_across_seeds']:.3f} | {stats['std_across_seeds']:.3f} |"
        )
    lines.extend(["", "## Episode Rewards", ""])
    for row in payload["rows"]:
        rewards = ";".join(f"{x:.3f}" for x in row["rewards"])
        lines.extend(
            [
                f"### seed{row['seed']} `{row['mode']}`",
                "",
                f"- mean/std/min/max: `{row['mean']:.3f}` / `{row['std']:.3f}` / `{row['min']:.3f}` / `{row['max']:.3f}`",
                f"- rewards: `{rewards}`",
                f"- rules loaded: `{row['rule_prior_num_rules']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--modes", nargs="+", default=["rule_off", "v7_current", "v8_safe"])
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "sd_retrospective_rule_induction" / "reports" / "hopper_v8_safe_eval_only_200k_v72_20260508",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT / "results" / "sd_retrospective_rule_induction" / "reports" / "hopper_v8_safe_eval_only_200k_v72_20260508.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.gpus) < len(args.seeds):
        raise ValueError("--gpus must provide at least one GPU per seed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for idx, seed in enumerate(args.seeds):
        gpu_id = args.gpus[idx]
        for mode in args.modes:
            run_eval(
                seed=seed,
                gpu_id=gpu_id,
                mode=mode,
                eval_episodes=args.eval_episodes,
                output_dir=args.output_dir,
                dry_run=args.dry_run,
                resume=args.resume,
            )
    if not args.dry_run:
        payload = summarize(args.output_dir, args.seeds, args.modes, args.summary)
        print(json.dumps(payload["by_mode"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
