#!/usr/bin/env python3
"""Run 10-episode rule-on/off eval sweep for v7.1a Hopper checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "experiments" / "sd_retrospective_rule_induction" / "eval_hopper_v7_rule_prior.py"
PYTHON = "/home/anaconda3/envs/sdmcts_p100/bin/python"
PREFIX = "tdmpc2_v71a_conf_closed_loop_5k_seed1_20260506"
RUN_ROOT = REPO_ROOT / "logs" / "hopper-hop" / "1"
CLOSED_LOOP_ROOT = (
    REPO_ROOT
    / "results"
    / "sd_retrospective_rule_induction"
    / "closed_loop"
    / "hopper_v71a_conf_100k_to150k_seed1_20260506"
)
CONFIDENCE_100K_RULES = (
    REPO_ROOT
    / "results"
    / "sd_retrospective_rule_induction"
    / "rules"
    / "hopper_v71a_100k_confidence_replay.json"
)
BASE_100K_CKPT = (
    REPO_ROOT
    / "logs"
    / "hopper-hop"
    / "1"
    / "tdmpc2_v71a_closed_loop_5k10_seed1_20260506_cycle020_target100019"
    / "models"
    / "final.full.pt"
)


def cycle_target(cycle: int) -> int:
    return cycle * 5000 + cycle - 1


def checkpoint_specs() -> list[dict[str, Any]]:
    specs = [
        {
            "label": "cycle020_step100020",
            "cycle": 20,
            "checkpoint": BASE_100K_CKPT,
            "rule_library": CONFIDENCE_100K_RULES,
        }
    ]
    for cycle in range(21, 31):
        target = cycle_target(cycle)
        specs.append(
            {
                "label": f"cycle{cycle:03d}_step{target}",
                "cycle": cycle,
                "checkpoint": RUN_ROOT / f"{PREFIX}_cycle{cycle:03d}_target{target}" / "models" / "final.full.pt",
                "rule_library": CLOSED_LOOP_ROOT / f"cycle_{cycle:03d}" / "rules" / f"library_closed_loop_cycle_{cycle:03d}.json",
            }
        )
    return specs


def parse_reward_list(text: str) -> list[float]:
    return [float(x) for x in str(text).split(";") if x != ""]


def run_eval(
    *,
    gpu_id: int,
    seed: int,
    eval_episodes: int,
    label: str,
    mode: str,
    checkpoint: Path,
    rule_library: Path | None,
    output_dir: Path,
    dry_run: bool,
) -> Path:
    report = output_dir / f"{label}_{mode}.json"
    exp_name = f"tdmpc2_v71a_10ep_sweep_{label}_{mode}_seed{seed}_20260506"
    cmd = [
        PYTHON,
        str(SCRIPT),
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
        f"+checkpoint={checkpoint}",
        "+eval_action_mode=eval",
        f"+eval_report={report}",
    ]
    if mode == "rule_on":
        if rule_library is None:
            raise ValueError("rule_on requires a rule library")
        cmd.extend(
            [
                "+rule_prior_enabled=true",
                f"+rule_prior_library={rule_library}",
                "+rule_prior_allowed_statuses=confidence_active",
                "+rule_prior_mode=penalty_only",
                "+rule_prior_opposite_penalty=0.02",
                "+rule_prior_preferred_cap=0.70",
                "+rule_prior_cap_penalty=0.02",
            ]
        )
    elif mode == "rule_off":
        cmd.append("+rule_prior_enabled=false")
    else:
        raise ValueError(f"unknown mode: {mode}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["MUJOCO_GL"] = "glfw"
    env["PYTHONPATH"] = f"{REPO_ROOT / 'external' / 'tdmpc2' / 'tdmpc2'}:{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    return report


def summarize(output_dir: Path, specs: list[dict[str, Any]], summary_path: Path) -> dict[str, Any]:
    rows = []
    for spec in specs:
        label = spec["label"]
        payloads = {}
        for mode in ("rule_off", "rule_on"):
            path = output_dir / f"{label}_{mode}.json"
            with path.open("r", encoding="utf-8") as f:
                payloads[mode] = json.load(f)
        off_rewards = parse_reward_list(payloads["rule_off"]["episode_reward_list"])
        on_rewards = parse_reward_list(payloads["rule_on"]["episode_reward_list"])
        paired = [on - off for on, off in zip(on_rewards, off_rewards)]
        rows.append(
            {
                "label": label,
                "cycle": spec["cycle"],
                "checkpoint": str(spec["checkpoint"]),
                "rule_library": str(spec["rule_library"]),
                "rule_off_mean": float(payloads["rule_off"]["episode_reward"]),
                "rule_on_mean": float(payloads["rule_on"]["episode_reward"]),
                "delta_on_minus_off": float(payloads["rule_on"]["episode_reward"] - payloads["rule_off"]["episode_reward"]),
                "rule_off_std": pstdev(off_rewards) if len(off_rewards) > 1 else 0.0,
                "rule_on_std": pstdev(on_rewards) if len(on_rewards) > 1 else 0.0,
                "paired_delta_mean": mean(paired) if paired else 0.0,
                "paired_delta_list": paired,
                "rule_off_list": off_rewards,
                "rule_on_list": on_rewards,
            }
        )
    payload = {"rows": rows}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    write_markdown(summary_path.with_suffix(".md"), rows)
    return payload


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Hopper v7.1a 10-Episode Rule On/Off Checkpoint Sweep",
        "",
        "| cycle | rule off mean | rule on mean | on-off | rule off std | rule on std |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['cycle']} | "
            f"{row['rule_off_mean']:.3f} | "
            f"{row['rule_on_mean']:.3f} | "
            f"{row['delta_on_minus_off']:.3f} | "
            f"{row['rule_off_std']:.3f} | "
            f"{row['rule_on_std']:.3f} |"
        )
    lines.extend(["", "## Episode Lists", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['label']}",
                "",
                f"- rule_off: `{';'.join(f'{x:.3f}' for x in row['rule_off_list'])}`",
                f"- rule_on: `{';'.join(f'{x:.3f}' for x in row['rule_on_list'])}`",
                f"- paired on-off: `{';'.join(f'{x:.3f}' for x in row['paired_delta_list'])}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "sd_retrospective_rule_induction" / "reports" / "hopper_v71a_10ep_rule_on_off_sweep_seed1_20260506")
    parser.add_argument("--summary", type=Path, default=REPO_ROOT / "results" / "sd_retrospective_rule_induction" / "reports" / "hopper_v71a_10ep_rule_on_off_sweep_seed1_20260506.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = checkpoint_specs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        checkpoint = Path(spec["checkpoint"])
        rule_library = Path(spec["rule_library"])
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        if not rule_library.exists():
            raise FileNotFoundError(rule_library)
        for mode in ("rule_off", "rule_on"):
            report = args.output_dir / f"{spec['label']}_{mode}.json"
            if report.exists() and not args.dry_run:
                print(f"skip existing report: {report}", flush=True)
                continue
            run_eval(
                gpu_id=args.gpu_id,
                seed=args.seed,
                eval_episodes=args.eval_episodes,
                label=spec["label"],
                mode=mode,
                checkpoint=checkpoint,
                rule_library=rule_library,
                output_dir=args.output_dir,
                dry_run=args.dry_run,
            )
    if not args.dry_run:
        summarize(args.output_dir, specs, args.summary)
        print(f"summary: {args.summary}", flush=True)


if __name__ == "__main__":
    main()
