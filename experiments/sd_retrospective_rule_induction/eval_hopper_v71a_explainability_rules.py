#!/usr/bin/env python3
"""Eval cycle029/030 Hopper checkpoints with explainability-v0 rules."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/anaconda3/envs/sdmcts_p100/bin/python"
EVAL_SCRIPT = REPO_ROOT / "experiments/sd_retrospective_rule_induction/eval_hopper_v7_rule_prior.py"
RUN_ROOT = REPO_ROOT / "logs/hopper-hop/1"
PREFIX = "tdmpc2_v71a_conf_closed_loop_5k_seed1_20260506"
EXPLAIN_RULES = (
    REPO_ROOT
    / "results/sd_retrospective_rule_induction/rules/hopper_v71a_explainability_v0_eval_rules_seed1_20260506.json"
)
BASE_SWEEP_DIR = (
    REPO_ROOT
    / "results/sd_retrospective_rule_induction/reports/hopper_v71a_10ep_rule_on_off_sweep_seed1_20260506"
)


def cycle_target(cycle: int) -> int:
    return cycle * 5000 + cycle - 1


def checkpoint(cycle: int) -> Path:
    target = cycle_target(cycle)
    return RUN_ROOT / f"{PREFIX}_cycle{cycle:03d}_target{target}" / "models/final.full.pt"


def parse_reward_list(text: str) -> list[float]:
    return [float(x) for x in str(text).split(";") if x != ""]


def run_eval(args: argparse.Namespace, cycle: int, dry_run: bool) -> Path:
    target = cycle_target(cycle)
    label = f"cycle{cycle:03d}_step{target}_explain_on"
    report = args.output_dir / f"{label}.json"
    exp_name = f"tdmpc2_v71a_10ep_explain_rules_{label}_seed{args.seed}_20260506"
    cmd = [
        PYTHON,
        str(EVAL_SCRIPT),
        "task=hopper-hop",
        "world_model_kind=tdmpc2",
        "model_size=null",
        "obs=state",
        f"seed={args.seed}",
        f"exp_name={exp_name}",
        "enable_wandb=false",
        "save_video=false",
        "save_agent=false",
        "save_csv=true",
        "save_full_state=false",
        "save_eval_traces=true",
        f"eval_episodes={args.eval_episodes}",
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
        f"+checkpoint={checkpoint(cycle)}",
        "+eval_action_mode=eval",
        f"+eval_report={report}",
        "+rule_prior_enabled=true",
        f"+rule_prior_library={args.rule_library}",
        "+rule_prior_allowed_statuses=explain_active",
        "+rule_prior_mode=penalty_only",
        "+rule_prior_opposite_penalty=0.02",
        "+rule_prior_preferred_cap=0.70",
        "+rule_prior_cap_penalty=0.02",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env["MUJOCO_GL"] = "glfw"
    env["PYTHONPATH"] = f"{REPO_ROOT / 'external' / 'tdmpc2' / 'tdmpc2'}:{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    return report


def load_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for cycle in args.cycles:
        target = cycle_target(cycle)
        base_label = f"cycle{cycle:03d}_step{target}"
        off = load_report(BASE_SWEEP_DIR / f"{base_label}_rule_off.json")
        conf = load_report(BASE_SWEEP_DIR / f"{base_label}_rule_on.json")
        expl = load_report(args.output_dir / f"{base_label}_explain_on.json")
        off_list = parse_reward_list(off["episode_reward_list"])
        conf_list = parse_reward_list(conf["episode_reward_list"])
        expl_list = parse_reward_list(expl["episode_reward_list"])
        rows.append(
            {
                "cycle": cycle,
                "target": target,
                "rule_off_mean": float(off["episode_reward"]),
                "confidence_on_mean": float(conf["episode_reward"]),
                "explain_on_mean": float(expl["episode_reward"]),
                "explain_minus_off": float(expl["episode_reward"] - off["episode_reward"]),
                "explain_minus_confidence": float(expl["episode_reward"] - conf["episode_reward"]),
                "rule_off_std": pstdev(off_list),
                "confidence_on_std": pstdev(conf_list),
                "explain_on_std": pstdev(expl_list),
                "rule_off_list": off_list,
                "confidence_on_list": conf_list,
                "explain_on_list": expl_list,
                "paired_explain_minus_off": [x - y for x, y in zip(expl_list, off_list)],
                "paired_explain_minus_confidence": [x - y for x, y in zip(expl_list, conf_list)],
            }
        )
    payload = {"rows": rows, "rule_library": str(args.rule_library)}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.summary.with_suffix(".md"), rows, args.rule_library)
    return payload


def write_markdown(path: Path, rows: list[dict[str, Any]], rule_library: Path) -> None:
    lines = [
        "# Hopper v7.1a Explainability Rule Eval",
        "",
        f"Rule library: `{rule_library}`",
        "",
        "| cycle | rule off | confidence on | explain on | explain-off | explain-confidence |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['cycle']} | {row['rule_off_mean']:.3f} | {row['confidence_on_mean']:.3f} | "
            f"{row['explain_on_mean']:.3f} | {row['explain_minus_off']:+.3f} | "
            f"{row['explain_minus_confidence']:+.3f} |"
        )
    lines += ["", "## Episode Lists", ""]
    for row in rows:
        lines += [
            f"### Cycle {row['cycle']:03d}",
            "",
            f"- rule_off: `{';'.join(f'{x:.3f}' for x in row['rule_off_list'])}`",
            f"- confidence_on: `{';'.join(f'{x:.3f}' for x in row['confidence_on_list'])}`",
            f"- explain_on: `{';'.join(f'{x:.3f}' for x in row['explain_on_list'])}`",
            f"- explain-rule_off: `{';'.join(f'{x:.3f}' for x in row['paired_explain_minus_off'])}`",
            "",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--cycles", type=int, nargs="+", default=[29, 30])
    parser.add_argument("--rule-library", type=Path, default=EXPLAIN_RULES)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/sd_retrospective_rule_induction/reports/hopper_v71a_explain_rules_eval_seed1_20260506",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT / "results/sd_retrospective_rule_induction/reports/hopper_v71a_explain_rules_eval_seed1_20260506.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.rule_library.exists():
        raise FileNotFoundError(args.rule_library)
    for cycle in args.cycles:
        report = args.output_dir / f"cycle{cycle:03d}_step{cycle_target(cycle)}_explain_on.json"
        if report.exists() and not args.dry_run:
            print(f"skip existing report: {report}", flush=True)
            continue
        run_eval(args, cycle, args.dry_run)
    if not args.dry_run:
        summarize(args)
        print(f"summary: {args.summary}", flush=True)


if __name__ == "__main__":
    main()
