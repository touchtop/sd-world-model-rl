#!/usr/bin/env python3
"""Eval-only sweep for gray-filtered Hopper rule-library variants.

This script does not train and does not write into the active closed-loop
experiment directory. It reuses the existing TD-MPC2 eval entry point and
stores all reports under ``results/gray_causal_rule_induction/eval_only``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = (
    REPO_ROOT
    / "experiments"
    / "sd_retrospective_rule_induction"
    / "eval_hopper_v7_rule_prior.py"
)
DEFAULT_PYTHON = "/home/anaconda3/envs/sdmcts_p100/bin/python"

RUN_PREFIX = (
    "tdmpc2_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_"
    "nostochgate_rngiso_seed123_resume300k_20260603_r1"
)
CLOSED_LOOP_ROOT = (
    REPO_ROOT
    / "results"
    / "sd_retrospective_rule_induction"
    / "closed_loop"
    / "hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_"
    "nostochgate_rngiso_seed123_resume300k_20260603_r1"
)
GRAY_RESULTS_ROOT = REPO_ROOT / "results" / "gray_causal_rule_induction"
VARIANT_SUFFIXES = [
    "drop_reject_active",
    "certified_active",
    "strict_active",
    "teacher_v8_active",
    "teacher_v8_certified_active",
]


def cycle_label(cycle: int) -> str:
    return f"cycle{cycle:03d}"


def cycle_target(cycle: int) -> int:
    """Return the target step convention used by the closed-loop launcher."""

    return cycle * 5000 + cycle - 1


def run_name_for_cycle(cycle: int) -> str:
    return f"{RUN_PREFIX}_{cycle_label(cycle)}_target{cycle_target(cycle)}"


def default_modes(rule_cycle: int) -> list[str]:
    prefix = cycle_label(rule_cycle)
    return [
        "rule_off",
        f"{prefix}_original_active",
        *[f"{prefix}_{suffix}" for suffix in VARIANT_SUFFIXES],
    ]


def cycle_root(cycle: int) -> Path:
    return CLOSED_LOOP_ROOT / f"cycle_{cycle:03d}"


def original_rule_library(rule_cycle: int) -> Path:
    return (
        cycle_root(rule_cycle)
        / "rules"
        / f"library_closed_loop_cycle_{rule_cycle:03d}_confidence.json"
    )


def sd_checkpoint(cycle: int) -> Path:
    return cycle_root(cycle) / "sd_explainers" / f"sd_explainer_cycle_{cycle:03d}.pt"


def tva_checkpoint(cycle: int) -> Path:
    return cycle_root(cycle) / "tva" / f"hopper_tva_cycle_{cycle:03d}.pt"


def variant_dir(rule_cycle: int) -> Path:
    return GRAY_RESULTS_ROOT / f"rule_library_variants_cycle{rule_cycle:03d}"


def normalize_mode(mode: str, rule_cycle: int) -> str:
    prefix = f"{cycle_label(rule_cycle)}_"
    if mode == "rule_off":
        return mode
    if mode.startswith(prefix):
        return mode.removeprefix(prefix)
    return mode


def display_mode(mode: str, rule_cycle: int) -> str:
    if mode == "rule_off":
        return mode
    prefix = f"{cycle_label(rule_cycle)}_"
    return mode if mode.startswith(prefix) else f"{prefix}{mode}"


def mode_library(mode: str, rule_cycle: int) -> Path | None:
    short = normalize_mode(mode, rule_cycle)
    prefix = cycle_label(rule_cycle)
    if short == "rule_off":
        return None
    if short == "original_active":
        return original_rule_library(rule_cycle)
    if short in VARIANT_SUFFIXES:
        return variant_dir(rule_cycle) / f"{prefix}_{short}.json"
    raise ValueError(f"unknown mode for cycle {rule_cycle}: {mode}")


def checkpoint(seed: int, args: argparse.Namespace) -> Path:
    run_name = args.checkpoint_run_name or run_name_for_cycle(args.checkpoint_cycle)
    return REPO_ROOT / "logs" / "hopper-hop" / str(seed) / run_name / "models" / "final.full.pt"


def parse_reward_list(text: Any) -> list[float]:
    if isinstance(text, list):
        return [float(x) for x in text]
    return [float(x) for x in str(text).split(";") if x != ""]


def guard_output_path(path: Path) -> None:
    if "sd_retrospective_rule_induction" in path.parts:
        raise ValueError(
            "Refusing to write gray eval outputs under sd_retrospective_rule_induction."
        )
    path.mkdir(parents=True, exist_ok=True)


def validate_inputs(args: argparse.Namespace) -> None:
    missing: list[Path] = []
    for seed in args.seeds:
        ckpt = checkpoint(seed, args)
        if not ckpt.exists():
            missing.append(ckpt)
    for mode in args.modes:
        library = mode_library(mode, args.rule_cycle)
        if library is not None and not library.exists():
            missing.append(library)
    if not args.disable_tva:
        for path in (sd_checkpoint(args.tva_cycle), tva_checkpoint(args.tva_cycle)):
            if not path.exists():
                missing.append(path)
    for mode in args.modes:
        if mode != "rule_off" and not sd_checkpoint(args.rule_cycle).exists():
            missing.append(sd_checkpoint(args.rule_cycle))
            break
    if missing:
        raise FileNotFoundError("\n".join(str(p) for p in missing))


def rule_args(mode: str, args: argparse.Namespace) -> list[str]:
    library = mode_library(mode, args.rule_cycle)
    if library is None:
        return ["+rule_prior_enabled=false"]
    return [
        "+rule_prior_enabled=true",
        f"+rule_prior_library={library}",
        "+rule_prior_allowed_statuses=confidence_active",
        "+rule_prior_mode=penalty_only",
        "+rule_prior_weight=0.05",
        "+rule_prior_opposite_penalty=0.01",
        "+rule_prior_preferred_cap=0.70",
        "+rule_prior_cap_penalty=0.01",
        "+rule_prior_v8_continue_score_min=0.60",
        f"+rule_prior_sd_checkpoint={sd_checkpoint(args.rule_cycle)}",
    ]


def tva_args(args: argparse.Namespace) -> list[str]:
    if args.disable_tva:
        return ["+tva_terminal_prior_enabled=false"]
    return [
        "+tva_terminal_prior_enabled=true",
        f"+tva_terminal_prior_sd_checkpoint={sd_checkpoint(args.tva_cycle)}",
        f"+tva_terminal_prior_checkpoint={tva_checkpoint(args.tva_cycle)}",
        "+tva_terminal_prior_weight=0.02",
        "+tva_terminal_prior_max_bonus=0.05",
        "+tva_terminal_prior_normalize=true",
    ]


def output_mode_name(mode: str, args: argparse.Namespace) -> str:
    return display_mode(mode, args.rule_cycle)


def build_cmd(
    *,
    args: argparse.Namespace,
    seed: int,
    mode: str,
) -> tuple[list[str], Path]:
    mode_name = output_mode_name(mode, args)
    ckpt = checkpoint(seed, args)
    report = args.output_dir / f"seed{seed}_{mode_name}.json"
    exp_name = (
        f"tdmpc2_gray_rule_eval_ckpt{args.checkpoint_cycle:03d}_"
        f"rules{args.rule_cycle:03d}_seed{seed}_{mode_name}_{args.eval_episodes}ep_20260604"
    )
    cmd = [
        args.python,
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
        f"+checkpoint={ckpt}",
        "+eval_action_mode=eval",
        f"+eval_report={report}",
        *rule_args(mode, args),
        *tva_args(args),
    ]
    return cmd, report


def launch_eval(
    *,
    args: argparse.Namespace,
    seed: int,
    gpu_id: int,
    mode: str,
) -> tuple[subprocess.Popen[bytes], Path, Path]:
    cmd, report = build_cmd(args=args, seed=seed, mode=mode)
    mode_name = output_mode_name(mode, args)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"seed{seed}_{mode_name}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["MUJOCO_GL"] = args.mujoco_gl
    env["PYTHONPATH"] = (
        f"{REPO_ROOT / 'external' / 'tdmpc2' / 'tdmpc2'}:{REPO_ROOT}:"
        f"{env.get('PYTHONPATH', '')}"
    )
    with log_path.open("wb") as log_file:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    print(
        f"launched seed={seed} gpu={gpu_id} mode={mode_name} report={report} log={log_path}",
        flush=True,
    )
    return proc, report, log_path


def wait_processes(processes: list[tuple[subprocess.Popen[bytes], Path, Path]]) -> None:
    failures: list[str] = []
    for proc, report, log_path in processes:
        code = proc.wait()
        if code != 0:
            failures.append(f"exit={code} report={report} log={log_path}")
    if failures:
        raise RuntimeError("eval process failed:\n" + "\n".join(failures))


def run_sweep(args: argparse.Namespace) -> None:
    if args.dry_run:
        for mode in args.modes:
            mode_name = output_mode_name(mode, args)
            for idx, seed in enumerate(args.seeds):
                gpu_id = args.gpus[idx]
                cmd, report = build_cmd(args=args, seed=seed, mode=mode)
                print(f"CUDA_VISIBLE_DEVICES={gpu_id} MUJOCO_GL={args.mujoco_gl} {' '.join(cmd)}")
                print(f"mode={mode_name} report={report}")
        return

    validate_inputs(args)
    guard_output_path(args.output_dir)
    for mode in args.modes:
        processes: list[tuple[subprocess.Popen[bytes], Path, Path]] = []
        mode_name = output_mode_name(mode, args)
        for idx, seed in enumerate(args.seeds):
            report = args.output_dir / f"seed{seed}_{mode_name}.json"
            if args.resume and report.exists():
                print(f"skip existing report: {report}", flush=True)
                continue
            processes.append(launch_eval(args=args, seed=seed, gpu_id=args.gpus[idx], mode=mode))
        wait_processes(processes)


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    mode_names = [output_mode_name(mode, args) for mode in args.modes]
    for mode_name in mode_names:
        for seed in args.seeds:
            path = args.output_dir / f"seed{seed}_{mode_name}.json"
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            rewards = parse_reward_list(payload.get("episode_reward_list", ""))
            rows.append(
                {
                    "seed": seed,
                    "mode": mode_name,
                    "mean": float(payload.get("episode_reward", 0.0)),
                    "std": pstdev(rewards) if len(rewards) > 1 else 0.0,
                    "min": min(rewards) if rewards else 0.0,
                    "max": max(rewards) if rewards else 0.0,
                    "rewards": rewards,
                    "rule_prior_enabled": bool(payload.get("rule_prior_enabled", False)),
                    "rule_prior_num_rules": int(payload.get("rule_prior_num_rules", 0)),
                    "tva_terminal_prior_bonus_mean": payload.get("tva_terminal_prior_bonus_mean"),
                    "tva_terminal_prior_bonus_max": payload.get("tva_terminal_prior_bonus_max"),
                    "report": str(path),
                }
            )

    by_mode: dict[str, dict[str, Any]] = {}
    for mode_name in mode_names:
        vals = [row["mean"] for row in rows if row["mode"] == mode_name]
        by_mode[mode_name] = {
            "seed_mean_values": vals,
            "mean_across_seeds": mean(vals) if vals else 0.0,
            "std_across_seeds": pstdev(vals) if len(vals) > 1 else 0.0,
            "mean_rule_prior_num_rules": mean(
                [row["rule_prior_num_rules"] for row in rows if row["mode"] == mode_name]
            )
            if vals
            else 0.0,
        }

    deltas: dict[str, dict[str, Any]] = {}
    original_mode = f"{cycle_label(args.rule_cycle)}_original_active"
    baselines = {
        "rule_off": {row["seed"]: row["mean"] for row in rows if row["mode"] == "rule_off"},
        original_mode: {row["seed"]: row["mean"] for row in rows if row["mode"] == original_mode},
    }
    for mode_name in mode_names:
        deltas[mode_name] = {}
        for baseline_name, baseline_values in baselines.items():
            if mode_name == baseline_name:
                continue
            vals = [
                row["mean"] - baseline_values[row["seed"]]
                for row in rows
                if row["mode"] == mode_name and row["seed"] in baseline_values
            ]
            deltas[mode_name][f"delta_vs_{baseline_name}_by_seed"] = vals
            deltas[mode_name][f"mean_delta_vs_{baseline_name}"] = mean(vals) if vals else 0.0

    run_name = args.checkpoint_run_name or run_name_for_cycle(args.checkpoint_cycle)
    libraries = {"rule_off": None}
    for mode in args.modes:
        mode_name = output_mode_name(mode, args)
        library = mode_library(mode, args.rule_cycle)
        libraries[mode_name] = str(library) if library is not None else None
    payload = {
        "run_name": run_name,
        "checkpoint_cycle": args.checkpoint_cycle,
        "rule_cycle": args.rule_cycle,
        "tva_cycle": args.tva_cycle,
        "checkpoint_family": f"cycle{args.checkpoint_cycle:03d} final.full.pt",
        "rule_libraries": libraries,
        "sd_checkpoint": str(sd_checkpoint(args.rule_cycle)),
        "tva_checkpoint": str(tva_checkpoint(args.tva_cycle)),
        "rows": rows,
        "by_mode": by_mode,
        "deltas": deltas,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    write_markdown(args.summary.with_suffix(".md"), payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    original_mode = f"cycle{int(payload['rule_cycle']):03d}_original_active"
    lines = [
        "# Gray Rule Variant Eval-only Sweep",
        "",
        f"Run: `{payload['run_name']}`",
        f"Checkpoint cycle: `{payload['checkpoint_cycle']}`",
        f"Rule cycle: `{payload['rule_cycle']}`",
        f"TVA cycle: `{payload['tva_cycle']}`",
        f"SD checkpoint: `{payload['sd_checkpoint']}`",
        f"TVA checkpoint: `{payload['tva_checkpoint']}`",
        "",
        "## Mode Summary",
        "",
        "| mode | seed means | mean | std | loaded rules |",
        "|---|---|---:|---:|---:|",
    ]
    for mode, stats in payload["by_mode"].items():
        seed_means = "; ".join(f"{x:.3f}" for x in stats["seed_mean_values"])
        lines.append(
            "| `{mode}` | `{seed_means}` | {mean:.3f} | {std:.3f} | {rules:.1f} |".format(
                mode=mode,
                seed_means=seed_means,
                mean=stats["mean_across_seeds"],
                std=stats["std_across_seeds"],
                rules=stats["mean_rule_prior_num_rules"],
            )
        )

    lines.extend(["", "## Deltas", "", "| mode | vs rule_off | vs original active |", "|---|---:|---:|"])
    for mode, stats in payload["deltas"].items():
        lines.append(
            "| `{mode}` | {off:.3f} | {orig:.3f} |".format(
                mode=mode,
                off=stats.get("mean_delta_vs_rule_off", 0.0),
                orig=stats.get(f"mean_delta_vs_{original_mode}", 0.0),
            )
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
                f"- report: `{row['report']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--modes", nargs="+")
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--rule-cycle", type=int, default=79)
    parser.add_argument("--tva-cycle", type=int)
    parser.add_argument("--checkpoint-cycle", type=int, default=80)
    parser.add_argument("--checkpoint-run-name")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--mujoco-gl", default="glfw")
    parser.add_argument("--disable-tva", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if args.tva_cycle is None:
        args.tva_cycle = args.rule_cycle
    if args.modes is None:
        args.modes = default_modes(args.rule_cycle)
    if args.output_dir is None:
        if args.rule_cycle == 79 and args.checkpoint_cycle == 80 and args.eval_episodes == 3:
            name = "cycle079_variants_on_400k_smoke3"
        else:
            name = (
                f"cycle{args.rule_cycle:03d}_variants_on_cycle{args.checkpoint_cycle:03d}_"
                f"{args.eval_episodes}ep"
            )
        args.output_dir = GRAY_RESULTS_ROOT / "eval_only" / name
    if args.summary is None:
        args.summary = args.output_dir.parent / f"{args.output_dir.name}_summary.json"
    return args


def main() -> None:
    args = parse_args()
    if len(args.gpus) < len(args.seeds):
        raise ValueError("--gpus must provide at least one GPU per seed")
    run_sweep(args)
    if not args.dry_run:
        payload = summarize(args)
        print(json.dumps(payload["by_mode"], ensure_ascii=False, indent=2, sort_keys=True))
        print(json.dumps(payload["deltas"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
