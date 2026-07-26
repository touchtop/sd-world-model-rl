#!/usr/bin/env python3
"""Run multi-seed Hopper closed loop with one shared retrospective rule library.

This entry keeps the research cadence separated from the evaluation cadence:
training is interrupted every 5k steps so that all seeds can contribute to one
shared SD explainer and one shared rule library, while formal checkpoint evals
remain sparse and are only triggered by the TD-MPC2 trainer every 25k steps.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from run_hopper_v7_closed_loop import (
    DEFAULT_PYTHON,
    REPO_ROOT,
    cycle_exp_name,
    cycle_paths,
    cycle_target_steps,
    summarize_cycle,
)


SCRIPT = REPO_ROOT / "scripts" / "run_hopper_v6_5k_rule_prior_smoke.sh"
EXPORT_REPLAY = REPO_ROOT / "experiments" / "sd_retrospective_rule_induction" / "export_hopper_replay_traces.py"
TVA_SD_INTERFACE_AUDIT = (
    REPO_ROOT
    / "experiments"
    / "sd_retrospective_rule_induction"
    / "audit_hopper_tva_sd_interface.py"
)


def parse_csv_ints(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def parse_initial_resume(text: str) -> tuple[int, Path]:
    if ":" not in text:
        raise argparse.ArgumentTypeError("expected SEED:CHECKPOINT")
    seed_text, path_text = text.split(":", 1)
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid seed in initial resume: {seed_text}") from exc
    path = Path(path_text)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"initial resume checkpoint not found: {path}")
    return seed, path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def parse_csv_horizons(text: str) -> list[int]:
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive horizons")
    return values


def make_tva_terminal_prior_policy(args: argparse.Namespace, mode: str) -> dict:
    if mode not in {"off", "ungated", "strong", "weak"}:
        raise ValueError(f"unknown TVA terminal prior policy mode: {mode}")
    allowed = mode in {"ungated", "strong", "weak"}
    if mode == "weak":
        weight = args.tva_terminal_weak_prior_weight
        max_bonus = args.tva_terminal_weak_prior_max_bonus
    else:
        weight = args.tva_terminal_prior_weight
        max_bonus = args.tva_terminal_prior_max_bonus
    return {
        "allowed": bool(allowed),
        "mode": mode,
        "weight": float(weight if allowed else 0.0),
        "max_bonus": float(max_bonus if allowed else 0.0),
    }


def ensure_initial_library(args: argparse.Namespace) -> Path:
    if args.initial_rule_library is not None:
        if not args.initial_rule_library.exists():
            raise FileNotFoundError(f"initial rule library not found: {args.initial_rule_library}")
        return args.initial_rule_library

    path = args.output_root / "shared_rules" / "library_cycle_000_empty.json"
    if not path.exists():
        status_meaning = {
            "cycle_candidate": "single-cycle evidence; not eligible for online use",
            "cycle_supported": "single-cycle supported evidence; requires cumulative validation",
            "replay_validated": "passed replay validation",
            "rolling_validated": "validated by repeated cycle evidence",
        }
        status_meaning.setdefault(args.rule_required_status, "required online-readable status for this run")
        write_json(
            path,
            {
                "library_version": "closed_loop_v7_multiseed_empty",
                "online_enabled": False,
                "status_meaning": status_meaning,
                "rules": [],
            },
        )
    return path


def build_train_env(
    args: argparse.Namespace,
    seed: int,
    gpu_id: int,
    cycle: int,
    rule_library: Path,
    sd_checkpoint: Path | None,
    tva_checkpoint: Path | None,
    resume_from: Path | None,
    tva_terminal_prior_policy: dict,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": args.python_bin,
            "WORLD_MODEL_KIND": "tdmpc2",
            "SD_ONLINE_SUPERVISION_ENABLED": "false",
            "STEPS": str(cycle_target_steps(args, cycle)),
            "EVAL_FREQ": str(args.formal_eval_freq),
            "CHECKPOINT_FREQ": str(args.formal_checkpoint_freq),
            "CHECKPOINT_AT_SEED": "false",
            "EVAL_EPISODES": str(args.formal_eval_episodes),
            "SEED_PRETRAIN_UPDATES": str(args.seed_pretrain_updates),
            "NUM_SAMPLES": str(args.num_samples),
            "NUM_ELITES": str(args.num_elites),
            "NUM_PI_TRAJS": str(args.num_pi_trajs),
            "ITERATIONS": str(args.iterations),
            "RULE_PRIOR_ENABLED": "true" if cycle > 1 or args.enable_rule_prior_cycle1 else "false",
            "RULE_PRIOR_MODE": args.rule_prior_mode,
            "RULE_PRIOR_WEIGHT": str(args.rule_prior_weight),
            "RULE_PRIOR_OPPOSITE_PENALTY": str(args.rule_prior_opposite_penalty),
            "RULE_PRIOR_PREFERRED_CAP": str(args.rule_prior_preferred_cap),
            "RULE_PRIOR_CAP_PENALTY": str(args.rule_prior_cap_penalty),
            "RULE_PRIOR_ALLOWED_STATUSES": args.rule_prior_allowed_statuses,
            "RULE_PRIOR_STOCHASTIC_GATE": "true" if args.enable_rule_prior_stochastic_gate else "false",
            "RULE_PRIOR_MIN_APPLY_PROB": str(args.rule_prior_min_apply_prob),
            "RULE_PRIOR_MAX_APPLY_PROB": str(args.rule_prior_max_apply_prob),
            "RULE_PRIOR_CONFIDENCE_PROB_POWER": str(args.rule_prior_confidence_prob_power),
            "RULE_PRIOR_STOCHASTIC_EVAL": "true" if args.rule_prior_stochastic_eval else "false",
            "RULE_PRIOR_STOCHASTIC_SEED": str(args.rule_prior_stochastic_seed + cycle * 1000 + seed),
            "RULE_PRIOR_PHASE_MATCH_MODE": args.rule_prior_phase_match_mode,
            "RULE_PRIOR_ENABLE_GROUNDED_HOP_PHASE": "true" if args.enable_rule_prior_grounded_hop_phase else "false",
            "RULE_LIBRARY": str(rule_library.resolve()),
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
        }
    )
    if (
        args.enable_tva_terminal_prior
        and bool(tva_terminal_prior_policy.get("allowed"))
        and sd_checkpoint is not None
        and tva_checkpoint is not None
    ):
        env.update(
            {
                "TVA_TERMINAL_PRIOR_ENABLED": "true",
                "TVA_TERMINAL_PRIOR_SD_CHECKPOINT": str(sd_checkpoint.resolve()),
                "TVA_TERMINAL_PRIOR_CHECKPOINT": str(tva_checkpoint.resolve()),
                "TVA_TERMINAL_PRIOR_WEIGHT": str(tva_terminal_prior_policy["weight"]),
                "TVA_TERMINAL_PRIOR_MAX_BONUS": str(tva_terminal_prior_policy["max_bonus"]),
                "TVA_TERMINAL_PRIOR_NORMALIZE": "true" if args.tva_terminal_prior_normalize else "false",
            }
        )
    else:
        env["TVA_TERMINAL_PRIOR_ENABLED"] = "false"
    if resume_from is not None:
        env["RESUME_FROM"] = str(resume_from.resolve())
    return env


def launch_seed_cycle(
    args: argparse.Namespace,
    seed: int,
    gpu_id: int,
    cycle: int,
    rule_library: Path,
    sd_checkpoint: Path | None,
    tva_checkpoint: Path | None,
    resume_from: Path | None,
    tva_terminal_prior_policy: dict,
) -> tuple[subprocess.Popen, Path, Path]:
    exp_name = cycle_exp_name(args, cycle)
    cmd = ["bash", str(SCRIPT), str(gpu_id), exp_name, str(seed)]
    env = build_train_env(
        args,
        seed,
        gpu_id,
        cycle,
        rule_library,
        sd_checkpoint,
        tva_checkpoint,
        resume_from,
        tva_terminal_prior_policy,
    )
    log_dir = args.output_root / f"cycle_{cycle:03d}" / "train_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"seed{seed}_gpu{gpu_id}.log"
    fp = log_path.open("w", encoding="utf-8")
    print(
        "\n".join(
            [
                f"[multiseed-cycle-{cycle:03d}] seed={seed} gpu={gpu_id}",
                f"[multiseed-cycle-{cycle:03d}] exp_name={exp_name}",
                f"[multiseed-cycle-{cycle:03d}] target_steps={cycle_target_steps(args, cycle)}",
                f"[multiseed-cycle-{cycle:03d}] rule_library={rule_library}",
                f"[multiseed-cycle-{cycle:03d}] tva_terminal_prior={args.enable_tva_terminal_prior}",
                f"[multiseed-cycle-{cycle:03d}] tva_terminal_prior_policy={tva_terminal_prior_policy}",
                f"[multiseed-cycle-{cycle:03d}] tva_sd_checkpoint={sd_checkpoint if sd_checkpoint else 'none'}",
                f"[multiseed-cycle-{cycle:03d}] tva_checkpoint={tva_checkpoint if tva_checkpoint else 'none'}",
                f"[multiseed-cycle-{cycle:03d}] resume_from={resume_from if resume_from else 'none'}",
                f"[multiseed-cycle-{cycle:03d}] log={log_path}",
            ]
        ),
        flush=True,
    )
    if args.dry_run:
        fp.write("DRY RUN: " + " ".join(cmd) + "\n")
        fp.close()
        return subprocess.Popen(["true"]), log_path, expected_checkpoint(args, seed, cycle)
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=fp, stderr=subprocess.STDOUT)
    return proc, log_path, expected_checkpoint(args, seed, cycle)


def expected_checkpoint(args: argparse.Namespace, seed: int, cycle: int) -> Path:
    return (
        REPO_ROOT
        / "logs"
        / "hopper-hop"
        / str(seed)
        / cycle_exp_name(args, cycle)
        / "models"
        / "final.full.pt"
    )


def run_seed_batch(
    args: argparse.Namespace,
    cycle: int,
    seeds: list[int],
    rule_library: Path,
    resumes: dict[int, Path | None],
    sd_checkpoint: Path | None,
    tva_checkpoint: Path | None,
    tva_terminal_prior_policy: dict,
) -> dict[int, Path]:
    next_resumes: dict[int, Path] = {}
    gpu_ids = args.gpu_ids
    for start in range(0, len(seeds), len(gpu_ids)):
        batch = seeds[start : start + len(gpu_ids)]
        running: list[tuple[int, subprocess.Popen, Path, Path]] = []
        for idx, seed in enumerate(batch):
            proc, log_path, checkpoint = launch_seed_cycle(
                args,
                seed=seed,
                gpu_id=gpu_ids[idx],
                cycle=cycle,
                rule_library=rule_library,
                sd_checkpoint=sd_checkpoint,
                tva_checkpoint=tva_checkpoint,
                resume_from=resumes.get(seed),
                tva_terminal_prior_policy=tva_terminal_prior_policy,
            )
            running.append((seed, proc, log_path, checkpoint))

        failed: list[tuple[int, int, Path]] = []
        for seed, proc, log_path, checkpoint in running:
            code = proc.wait()
            if code != 0:
                failed.append((seed, code, log_path))
            elif not args.dry_run and not checkpoint.exists():
                failed.append((seed, 99, log_path))
            else:
                next_resumes[seed] = checkpoint

        if failed:
            for _, proc, _, _ in running:
                if proc.poll() is None:
                    proc.terminate()
            detail = "; ".join(f"seed={seed} code={code} log={log}" for seed, code, log in failed)
            raise RuntimeError(f"training batch failed: {detail}")
    return next_resumes


def export_replay_traces(args: argparse.Namespace, cycle: int, resumes: dict[int, Path]) -> None:
    for seed, checkpoint in sorted(resumes.items()):
        run_dir = checkpoint.parents[1]
        cmd = [
            args.python_bin,
            str(EXPORT_REPLAY),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(run_dir / "retrospective_train_traces"),
            "--metadata",
            str(args.output_root / f"cycle_{cycle:03d}" / "train_replay_exports" / f"seed{seed}.json"),
        ]
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def write_cycle_report(
    args: argparse.Namespace,
    cycle: int,
    rule_library: Path,
    prev_sd_checkpoint: Path | None,
    prev_tva_checkpoint: Path | None,
    resumes: dict[int, Path | None],
    tva_terminal_prior_policy: dict,
    tva_terminal_prior_gate_report: Path | None,
) -> None:
    payload = {
        "cycle": cycle,
        "target_steps": cycle_target_steps(args, cycle),
        "seeds": args.seeds,
        "formal_eval_freq": args.formal_eval_freq,
        "formal_checkpoint_freq": args.formal_checkpoint_freq,
        "formal_eval_episodes": args.formal_eval_episodes,
        "shared_rule_library_before_cycle": str(rule_library),
        "shared_sd_checkpoint_before_cycle": str(prev_sd_checkpoint) if prev_sd_checkpoint else None,
        "shared_tva_checkpoint_before_cycle": str(prev_tva_checkpoint) if prev_tva_checkpoint else None,
        "tva_terminal_prior_requested": bool(args.enable_tva_terminal_prior),
        "tva_terminal_prior_gate_enabled": bool(args.enable_tva_terminal_prior_gate),
        "tva_terminal_prior_allowed_this_cycle": bool(tva_terminal_prior_policy.get("allowed")),
        "tva_terminal_prior_policy_this_cycle": tva_terminal_prior_policy,
        "tva_terminal_prior_gate_report": str(tva_terminal_prior_gate_report) if tva_terminal_prior_gate_report else None,
        "resume_checkpoints": {str(k): str(v) if v else None for k, v in resumes.items()},
    }
    write_json(args.output_root / f"cycle_{cycle:03d}" / "cycle_config.json", payload)


def tva_gate_report_path(args: argparse.Namespace, cycle: int) -> Path:
    return args.output_root / f"cycle_{cycle:03d}" / "reports" / "tva_terminal_prior_gate.json"


def tva_interface_audit_path(args: argparse.Namespace, cycle: int) -> Path:
    return args.output_root / f"cycle_{cycle:03d}" / "reports" / "tva_sd_interface_audit_gate.json"


def evaluate_tva_terminal_gate(args: argparse.Namespace, audit_path: Path) -> dict:
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    weak_failures: list[str] = []
    horizon_results: dict[str, dict] = {}
    for horizon in args.tva_terminal_gate_horizons:
        hkey = str(horizon)
        summary = payload.get("horizon_summary", {}).get(hkey)
        if not summary:
            failures.append(f"h={horizon}: missing horizon summary")
            weak_failures.append(f"h={horizon}: missing horizon summary")
            continue
        oracle = summary.get("oracle_terminal", {})
        sd = summary.get("sd_terminal", {})
        abs_delta = summary.get("abs_value_delta", {})
        n = int(sd.get("n", 0))
        sd_rmse = float(sd.get("rmse", 0.0))
        baseline_rmse = float(sd.get("baseline_rmse", 0.0))
        rmse_ratio = sd_rmse / baseline_rmse if baseline_rmse > 1e-12 else float("inf")
        result = {
            "n": n,
            "oracle_pearson": float(oracle.get("pearson", 0.0)),
            "sd_pearson": float(sd.get("pearson", 0.0)),
            "sd_r2": float(sd.get("r2", 0.0)),
            "sd_rmse": sd_rmse,
            "sd_baseline_rmse": baseline_rmse,
            "sd_rmse_ratio": rmse_ratio,
            "mean_abs_delta_v": float(abs_delta.get("mean", 0.0)),
        }
        strong_horizon_failures: list[str] = []
        weak_horizon_failures: list[str] = []
        if n < args.tva_terminal_gate_min_samples:
            strong_horizon_failures.append(f"n {n} < {args.tva_terminal_gate_min_samples}")
        if result["oracle_pearson"] < args.tva_terminal_gate_min_oracle_pearson:
            strong_horizon_failures.append(
                f"oracle pearson {result['oracle_pearson']:.3f} < {args.tva_terminal_gate_min_oracle_pearson:.3f}"
            )
        if result["sd_pearson"] < args.tva_terminal_gate_min_sd_pearson:
            strong_horizon_failures.append(
                f"sd pearson {result['sd_pearson']:.3f} < {args.tva_terminal_gate_min_sd_pearson:.3f}"
            )
        if result["sd_r2"] < args.tva_terminal_gate_min_sd_r2:
            strong_horizon_failures.append(f"sd R2 {result['sd_r2']:.3f} < {args.tva_terminal_gate_min_sd_r2:.3f}")
        if rmse_ratio > args.tva_terminal_gate_max_sd_rmse_ratio:
            strong_horizon_failures.append(
                f"sd RMSE ratio {rmse_ratio:.3f} > {args.tva_terminal_gate_max_sd_rmse_ratio:.3f}"
            )
        if (
            args.tva_terminal_gate_max_abs_delta_v > 0
            and result["mean_abs_delta_v"] > args.tva_terminal_gate_max_abs_delta_v
        ):
            strong_horizon_failures.append(
                f"mean abs(delta V) {result['mean_abs_delta_v']:.3f} > {args.tva_terminal_gate_max_abs_delta_v:.3f}"
            )

        if n < args.tva_terminal_weak_gate_min_samples:
            weak_horizon_failures.append(f"n {n} < {args.tva_terminal_weak_gate_min_samples}")
        if result["oracle_pearson"] < args.tva_terminal_weak_gate_min_oracle_pearson:
            weak_horizon_failures.append(
                f"oracle pearson {result['oracle_pearson']:.3f} < {args.tva_terminal_weak_gate_min_oracle_pearson:.3f}"
            )
        if result["sd_pearson"] < args.tva_terminal_weak_gate_min_sd_pearson:
            weak_horizon_failures.append(
                f"sd pearson {result['sd_pearson']:.3f} < {args.tva_terminal_weak_gate_min_sd_pearson:.3f}"
            )
        if result["sd_r2"] < args.tva_terminal_weak_gate_min_sd_r2:
            weak_horizon_failures.append(
                f"sd R2 {result['sd_r2']:.3f} < {args.tva_terminal_weak_gate_min_sd_r2:.3f}"
            )
        if rmse_ratio > args.tva_terminal_weak_gate_max_sd_rmse_ratio:
            weak_horizon_failures.append(
                f"sd RMSE ratio {rmse_ratio:.3f} > {args.tva_terminal_weak_gate_max_sd_rmse_ratio:.3f}"
            )

        result["strong_passed"] = not strong_horizon_failures
        result["strong_failures"] = strong_horizon_failures
        result["weak_passed"] = not weak_horizon_failures
        result["weak_failures"] = weak_horizon_failures
        horizon_results[hkey] = result
        failures.extend(f"h={horizon}: {x}" for x in strong_horizon_failures)
        weak_failures.extend(f"h={horizon}: {x}" for x in weak_horizon_failures)
    strong_passed = not failures
    weak_horizon_passes = [r.get("weak_passed", False) for r in horizon_results.values()]
    if args.enable_tva_terminal_prior_weak_gate and weak_horizon_passes:
        if args.tva_terminal_weak_gate_horizon_mode == "all":
            weak_passed = all(weak_horizon_passes)
        else:
            weak_passed = any(weak_horizon_passes)
    else:
        weak_passed = False
    if strong_passed:
        decision_mode = "strong"
    elif weak_passed:
        decision_mode = "weak"
    else:
        decision_mode = "off"
    policy = make_tva_terminal_prior_policy(args, decision_mode)
    return {
        "gate_family": "tva_terminal_prior_interface_gate_v2",
        "audit": str(audit_path),
        "passed": bool(policy["allowed"]),
        "strong_passed": bool(strong_passed),
        "weak_passed": bool(weak_passed),
        "decision": {
            "mode": decision_mode,
            "allowed": bool(policy["allowed"]),
            "weight": policy["weight"],
            "max_bonus": policy["max_bonus"],
        },
        "failures": failures,
        "weak_failures": weak_failures,
        "horizons": horizon_results,
        "thresholds": {
            "strong": {
                "horizons": args.tva_terminal_gate_horizons,
                "min_samples": args.tva_terminal_gate_min_samples,
                "min_oracle_pearson": args.tva_terminal_gate_min_oracle_pearson,
                "min_sd_pearson": args.tva_terminal_gate_min_sd_pearson,
                "min_sd_r2": args.tva_terminal_gate_min_sd_r2,
                "max_sd_rmse_ratio": args.tva_terminal_gate_max_sd_rmse_ratio,
                "max_abs_delta_v": args.tva_terminal_gate_max_abs_delta_v,
            },
            "weak": {
                "enabled": args.enable_tva_terminal_prior_weak_gate,
                "horizon_mode": args.tva_terminal_weak_gate_horizon_mode,
                "min_samples": args.tva_terminal_weak_gate_min_samples,
                "min_oracle_pearson": args.tva_terminal_weak_gate_min_oracle_pearson,
                "min_sd_pearson": args.tva_terminal_weak_gate_min_sd_pearson,
                "min_sd_r2": args.tva_terminal_weak_gate_min_sd_r2,
                "max_sd_rmse_ratio": args.tva_terminal_weak_gate_max_sd_rmse_ratio,
                "weight": args.tva_terminal_weak_prior_weight,
                "max_bonus": args.tva_terminal_weak_prior_max_bonus,
            },
        },
    }


def run_tva_terminal_gate(
    args: argparse.Namespace,
    cycle: int,
    selection: Path,
    sd_checkpoint: Path | None,
    tva_checkpoint: Path | None,
) -> tuple[dict, Path | None]:
    if not args.enable_tva_terminal_prior_gate:
        return make_tva_terminal_prior_policy(args, "ungated"), None
    report_path = tva_gate_report_path(args, cycle)
    if sd_checkpoint is None or tva_checkpoint is None:
        policy = make_tva_terminal_prior_policy(args, "off")
        report = {
            "gate_family": "tva_terminal_prior_interface_gate_v2",
            "passed": False,
            "strong_passed": False,
            "weak_passed": False,
            "decision": {
                "mode": "off",
                "allowed": False,
                "weight": policy["weight"],
                "max_bonus": policy["max_bonus"],
            },
            "failures": ["missing SD or TVA checkpoint"],
            "weak_failures": ["missing SD or TVA checkpoint"],
            "audit": None,
        }
        write_json(report_path, report)
        return policy, report_path
    audit_path = tva_interface_audit_path(args, cycle)
    cmd = [
        args.python_bin,
        str(TVA_SD_INTERFACE_AUDIT),
        "--selection", str(selection),
        "--sd-checkpoint", str(sd_checkpoint),
        "--tva-checkpoint", str(tva_checkpoint),
        "--output", str(audit_path),
        "--horizons", ",".join(str(h) for h in args.tva_terminal_gate_horizons),
        "--max-episodes", str(args.tva_terminal_gate_max_episodes),
        "--max-starts-per-episode", str(args.tva_terminal_gate_max_starts_per_episode),
        "--seed", str(args.selection_seed + cycle),
        "--gamma", str(args.tva_gamma),
        "--true-coef", str(args.tva_true_coef),
        "--shaped-coef", str(args.tva_shaped_coef),
        "--device", args.tva_device,
    ]
    print(" ".join(cmd), flush=True)
    if not args.dry_run:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        report = evaluate_tva_terminal_gate(args, audit_path)
    else:
        policy = make_tva_terminal_prior_policy(args, "off")
        report = {
            "gate_family": "tva_terminal_prior_interface_gate_v2",
            "passed": False,
            "strong_passed": False,
            "weak_passed": False,
            "decision": {
                "mode": "off",
                "allowed": False,
                "weight": policy["weight"],
                "max_bonus": policy["max_bonus"],
            },
            "failures": ["dry run"],
            "weak_failures": ["dry run"],
            "audit": str(audit_path),
        }
    write_json(report_path, report)
    policy = make_tva_terminal_prior_policy(args, report.get("decision", {}).get("mode", "off"))
    print(
        f"[multiseed-cycle-{cycle:03d}] tva_terminal_prior_gate_passed={report['passed']} "
        f"mode={policy['mode']} weight={policy['weight']} max_bonus={policy['max_bonus']} report={report_path}",
        flush=True,
    )
    if report.get("failures"):
        print(f"[multiseed-cycle-{cycle:03d}] tva_terminal_prior_gate_failures={report['failures']}", flush=True)
    if report.get("weak_failures") and policy["mode"] == "off":
        print(f"[multiseed-cycle-{cycle:03d}] tva_terminal_prior_weak_gate_failures={report['weak_failures']}", flush=True)
    return policy, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=parse_csv_ints, default=[1, 2, 3])
    parser.add_argument("--gpu-ids", type=parse_csv_ints, required=True)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--start-cycle", type=int, default=1)
    parser.add_argument("--cycle-steps", type=int, default=5000)
    parser.add_argument("--align-resumed-eval", action="store_true", default=True)
    parser.add_argument("--exp-prefix", default="tdmpc2_v72_multiseed_shared_5k_seed123_20260507")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "sd_retrospective_rule_induction"
        / "closed_loop"
        / "hopper_v72_multiseed_shared_100k_seed123_20260507",
    )
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON)
    parser.add_argument("--initial-rule-library", type=Path, default=None)
    parser.add_argument("--initial-sd-checkpoint", type=Path, default=None)
    parser.add_argument("--initial-tva-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--initial-tva-terminal-prior-policy",
        choices=["auto", "off", "ungated", "strong", "weak"],
        default="auto",
        help="Initial TVA terminal prior policy for resumed gated runs before the first new cycle summary.",
    )
    parser.add_argument(
        "--initial-resume",
        action="append",
        type=parse_initial_resume,
        default=[],
        help="Initial TD-MPC2 resume checkpoint as SEED:PATH; may be repeated.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enable-rule-prior-cycle1", action="store_true")

    parser.add_argument("--formal-eval-freq", type=int, default=25000)
    parser.add_argument("--formal-checkpoint-freq", type=int, default=25000)
    parser.add_argument("--formal-eval-episodes", type=int, default=10)
    parser.add_argument("--seed-pretrain-updates", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--num-elites", type=int, default=8)
    parser.add_argument("--num-pi-trajs", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=3)

    parser.add_argument("--rule-prior-mode", default="penalty_only")
    parser.add_argument("--rule-prior-weight", type=float, default=0.05)
    parser.add_argument("--rule-prior-opposite-penalty", type=float, default=0.01)
    parser.add_argument("--rule-prior-preferred-cap", type=float, default=0.70)
    parser.add_argument("--rule-prior-cap-penalty", type=float, default=0.01)
    parser.add_argument("--rule-prior-allowed-statuses", default="replay_validated,rolling_validated")
    parser.add_argument("--enable-rule-prior-stochastic-gate", action="store_true", default=False)
    parser.add_argument("--rule-prior-min-apply-prob", type=float, default=0.40)
    parser.add_argument("--rule-prior-max-apply-prob", type=float, default=0.98)
    parser.add_argument("--rule-prior-confidence-prob-power", type=float, default=1.0)
    parser.add_argument("--rule-prior-stochastic-eval", action="store_true", default=False)
    parser.add_argument(
        "--rule-prior-stochastic-seed",
        type=int,
        default=20260528,
        help="Base seed for the rule-prior stochastic gate private RNG.",
    )
    parser.add_argument(
        "--rule-prior-phase-match-mode",
        choices=["current", "hop_alias"],
        default="current",
        help="State phase matching mode for teacher/core substate action rules.",
    )
    parser.add_argument(
        "--enable-rule-prior-grounded-hop-phase",
        action="store_true",
        default=False,
        help="Allow grounded_hop to override prone in online rule-prior phase matching.",
    )
    parser.add_argument("--rule-required-status", default="replay_validated")
    parser.add_argument("--enable-tva-terminal-prior", action="store_true", default=False)
    parser.add_argument("--tva-terminal-prior-weight", type=float, default=0.02)
    parser.add_argument("--tva-terminal-prior-max-bonus", type=float, default=0.05)
    parser.add_argument("--tva-terminal-prior-normalize", action="store_true", default=True)
    parser.add_argument(
        "--enable-tva-terminal-prior-weak-gate",
        action="store_true",
        default=False,
        help="Allow a low-weight TVA terminal prior when only directional interface evidence passes.",
    )
    parser.add_argument("--tva-terminal-weak-prior-weight", type=float, default=0.005)
    parser.add_argument("--tva-terminal-weak-prior-max-bonus", type=float, default=0.01)
    parser.add_argument(
        "--enable-tva-terminal-prior-gate",
        action="store_true",
        default=False,
        help="Enable TVA terminal prior only when the previous cycle passes TVA-SD interface audit thresholds.",
    )
    parser.add_argument("--tva-terminal-gate-horizons", type=parse_csv_horizons, default=[1, 3])
    parser.add_argument("--tva-terminal-gate-max-episodes", type=int, default=20)
    parser.add_argument("--tva-terminal-gate-max-starts-per-episode", type=int, default=2)
    parser.add_argument("--tva-terminal-gate-min-samples", type=int, default=30)
    parser.add_argument("--tva-terminal-gate-min-oracle-pearson", type=float, default=0.55)
    parser.add_argument("--tva-terminal-gate-min-sd-pearson", type=float, default=0.55)
    parser.add_argument("--tva-terminal-gate-min-sd-r2", type=float, default=0.20)
    parser.add_argument("--tva-terminal-gate-max-sd-rmse-ratio", type=float, default=0.90)
    parser.add_argument(
        "--tva-terminal-gate-max-abs-delta-v",
        type=float,
        default=0.0,
        help="Optional cap on mean |V(SD terminal)-V(true terminal)|; non-positive disables this threshold.",
    )
    parser.add_argument("--tva-terminal-weak-gate-horizon-mode", choices=["any", "all"], default="any")
    parser.add_argument("--tva-terminal-weak-gate-min-samples", type=int, default=30)
    parser.add_argument("--tva-terminal-weak-gate-min-oracle-pearson", type=float, default=0.50)
    parser.add_argument("--tva-terminal-weak-gate-min-sd-pearson", type=float, default=0.50)
    parser.add_argument("--tva-terminal-weak-gate-min-sd-r2", type=float, default=0.0)
    parser.add_argument("--tva-terminal-weak-gate-max-sd-rmse-ratio", type=float, default=1.0)

    parser.add_argument("--summary-window-cycles", type=int, default=3)
    parser.add_argument("--trajectory-score-mode", choices=["reward", "stage", "hybrid"], default="stage")
    parser.add_argument("--max-online-rules-per-cycle", type=int, default=2)
    parser.add_argument("--max-cycle-candidates", type=int, default=12)
    parser.add_argument("--rolling-min-cycles", type=int, default=2)
    parser.add_argument("--rolling-min-support", type=int, default=6)
    parser.add_argument("--rolling-min-advantage", type=float, default=0.05)
    parser.add_argument("--rolling-max-good-share", type=float, default=0.25)
    parser.add_argument("--rolling-weaken-max-good-share", type=float, default=0.40)
    parser.add_argument("--sd-epochs", type=int, default=3)
    parser.add_argument("--sd-max-transitions", type=int, default=50000)
    parser.add_argument("--sd-batch-size", type=int, default=512)
    parser.add_argument("--sd-lr", type=float, default=5e-4)
    parser.add_argument("--sd-device", default="cpu")
    parser.add_argument("--sd-transition-residual-scale", type=float, default=1.0)
    parser.add_argument("--enable-tva-audit", action="store_true", default=False)
    parser.add_argument("--tva-epochs", type=int, default=50)
    parser.add_argument("--tva-max-transitions", type=int, default=50000)
    parser.add_argument("--tva-hidden", type=int, default=128)
    parser.add_argument("--tva-batch-size", type=int, default=256)
    parser.add_argument("--tva-lr", type=float, default=1e-3)
    parser.add_argument("--tva-gamma", type=float, default=0.99)
    parser.add_argument("--tva-true-coef", type=float, default=1.0)
    parser.add_argument("--tva-shaped-coef", type=float, default=1.0)
    parser.add_argument("--tva-device", default="cpu")
    parser.add_argument("--tva-audit-max-records", type=int, default=300)
    parser.add_argument("--tva-audit-sample-mode", choices=["first", "stratified", "label_event"], default="stratified")
    parser.add_argument("--enable-tva-shadow-update", action="store_true", default=False)
    parser.add_argument("--tva-shadow-family-top-k", type=int, default=2)
    parser.add_argument("--tva-shadow-family-advantage-threshold", type=float, default=0.05)
    parser.add_argument("--tva-shadow-min-hits", type=int, default=20)
    parser.add_argument("--tva-shadow-min-effect-hit-ratio", type=float, default=0.003)
    parser.add_argument("--tva-shadow-min-positive-rate", type=float, default=0.10)
    parser.add_argument("--tva-shadow-min-opposite-rate", type=float, default=0.10)
    parser.add_argument("--tva-shadow-min-directional-rate-margin", type=float, default=0.05)
    parser.add_argument("--tva-shadow-max-best-family-entropy", type=float, default=0.65)
    parser.add_argument("--tva-shadow-target-opposite-margin", type=float, default=0.50)
    parser.add_argument("--tva-shadow-hit-ref", type=float, default=300.0)
    parser.add_argument("--tva-shadow-positive-lr", type=float, default=0.25)
    parser.add_argument("--tva-shadow-harmful-lr", type=float, default=0.70)
    parser.add_argument("--tva-shadow-advantage-scale", type=float, default=20.0)
    parser.add_argument("--enable-tva-consequence-gate", action="store_true", default=False)
    parser.add_argument("--tva-consequence-horizon", type=int, default=5)
    parser.add_argument("--tva-consequence-action-threshold", type=float, default=0.20)
    parser.add_argument("--tva-consequence-min-outcome-margin", type=float, default=0.0)
    parser.add_argument("--tva-consequence-min-relation-count", type=int, default=10)
    parser.add_argument("--max-states-per-episode", type=int, default=10)
    parser.add_argument("--counterfactual-horizon", type=int, default=5)
    parser.add_argument("--postprocess-workers", type=int, default=0)
    parser.add_argument("--good-quantile", type=float, default=0.80)
    parser.add_argument("--bad-quantile", type=float, default=0.20)
    parser.add_argument("--random-per-cycle", type=int, default=10)
    parser.add_argument("--selection-memory-policy", type=Path, default=None)
    parser.add_argument("--max-selected-episodes", type=int, default=None)
    parser.add_argument("--max-selected-per-cycle-step", type=int, default=None)
    parser.add_argument("--selection-preserve-sources", action="append", default=None)
    parser.add_argument("--selection-seed", type=int, default=20260507)
    parser.add_argument("--enable-v8-transition-rules", action="store_true", default=False)
    parser.add_argument("--v8-success-min-length", type=int, default=3)
    parser.add_argument("--v8-success-lookahead", type=int, default=25)
    parser.add_argument("--v8-transition-min-support", type=int, default=5)
    parser.add_argument("--v8-transition-min-seeds", type=int, default=2)
    parser.add_argument("--v8-counterfactual-max-records-per-key", type=int, default=40)
    parser.add_argument("--v8-counterfactual-min-support", type=int, default=5)
    parser.add_argument("--v8-counterfactual-min-advantage", type=float, default=0.02)
    parser.add_argument("--v8-validation-min-support", type=int, default=2)
    parser.add_argument("--v8-validation-min-seeds", type=int, default=1)
    parser.add_argument("--v8-validation-min-advantage", type=float, default=0.02)
    parser.add_argument("--v8-safety-max-episodes-per-seed", type=int, default=0)
    parser.add_argument("--v8-max-false-intervention-rate", type=float, default=0.10)
    parser.add_argument("--v8-max-online-transition-rules-per-cycle", type=int, default=2)
    parser.add_argument("--v8-max-cycle-transition-candidates", type=int, default=12)
    parser.add_argument("--enable-rule-effect-confidence", action="store_true", default=False)
    parser.add_argument("--rule-effect-high-episode-reward", type=float, default=100.0)
    parser.add_argument("--rule-effect-continue-score-min", type=float, default=0.60)
    parser.add_argument("--rule-effect-action-threshold", type=float, default=0.20)
    parser.add_argument("--confidence-default", type=float, default=0.56)
    parser.add_argument("--confidence-active-threshold", type=float, default=0.55)
    parser.add_argument("--confidence-retired-threshold", type=float, default=0.05)
    parser.add_argument("--confidence-min-effect-hits", type=int, default=20)
    parser.add_argument("--confidence-hit-ref", type=float, default=500.0)
    parser.add_argument("--confidence-positive-threshold", type=float, default=0.75)
    parser.add_argument("--confidence-harmful-threshold", type=float, default=0.45)
    parser.add_argument("--confidence-positive-lr", type=float, default=0.25)
    parser.add_argument("--confidence-harmful-lr", type=float, default=0.90)
    parser.add_argument("--confidence-absence-decay", type=float, default=0.98)
    parser.add_argument("--confidence-low-hit-decay", type=float, default=0.99)
    parser.add_argument("--confidence-context-floor", type=float, default=0.50)
    parser.add_argument("--confidence-positive-context-threshold", type=float, default=0.0)
    parser.add_argument("--confidence-hard-harmful-threshold", type=float, default=0.50)
    parser.add_argument("--confidence-hard-harmful-min-hits", type=int, default=100)
    parser.add_argument("--confidence-hard-harmful-margin", type=float, default=0.05)
    parser.add_argument("--confidence-conflict-keep-confidence-margin", type=float, default=0.15)
    parser.add_argument("--confidence-conflict-keep-harmful-margin", type=float, default=0.10)
    parser.add_argument("--confidence-conflict-demote-margin", type=float, default=0.05)
    parser.add_argument(
        "--protected-teacher-origins",
        default="teacher_core_standard_hop",
        help="Comma-separated rule_origin values protected from weak conflict demotion.",
    )
    parser.add_argument(
        "--disable-teacher-core-conflict-protection",
        action="store_true",
        default=False,
        help="Disable teacher-core conflict protection in rule-effect confidence replay.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.cycles <= 0 or args.start_cycle <= 0 or args.cycle_steps <= 0:
        raise ValueError("cycles, start-cycle, and cycle-steps must be positive.")
    if args.start_cycle > args.cycles:
        raise ValueError("start-cycle must be <= cycles.")
    if args.formal_eval_freq % args.cycle_steps != 0:
        raise ValueError("formal eval frequency should be an integer multiple of cycle steps.")
    if args.formal_checkpoint_freq != args.formal_eval_freq:
        raise ValueError("this experiment expects checkpoint and formal eval frequencies to match.")
    if args.enable_tva_terminal_prior_gate and not args.enable_tva_terminal_prior:
        raise ValueError("--enable-tva-terminal-prior-gate requires --enable-tva-terminal-prior.")
    if args.enable_tva_terminal_prior_gate and not args.enable_tva_audit:
        raise ValueError("--enable-tva-terminal-prior-gate requires --enable-tva-audit so TVA checkpoints are available.")
    if args.enable_tva_terminal_prior_weak_gate and not args.enable_tva_terminal_prior_gate:
        raise ValueError("--enable-tva-terminal-prior-weak-gate requires --enable-tva-terminal-prior-gate.")
    if args.tva_terminal_prior_weight < 0 or args.tva_terminal_prior_max_bonus < 0:
        raise ValueError("TVA terminal prior weight and max bonus must be non-negative.")
    if args.tva_terminal_weak_prior_weight < 0 or args.tva_terminal_weak_prior_max_bonus < 0:
        raise ValueError("weak TVA terminal prior weight and max bonus must be non-negative.")
    if args.tva_terminal_weak_prior_weight > args.tva_terminal_prior_weight:
        raise ValueError("weak TVA terminal prior weight should not exceed the strong prior weight.")
    if args.tva_terminal_weak_prior_max_bonus > args.tva_terminal_prior_max_bonus:
        raise ValueError("weak TVA terminal prior max bonus should not exceed the strong prior max bonus.")
    if not (0.0 <= args.rule_prior_min_apply_prob <= args.rule_prior_max_apply_prob <= 1.0):
        raise ValueError("rule prior apply probabilities must satisfy 0 <= min <= max <= 1.")
    if args.rule_prior_confidence_prob_power <= 0.0:
        raise ValueError("rule_prior_confidence_prob_power must be positive.")
    if args.tva_terminal_gate_min_samples <= 0:
        raise ValueError("tva terminal gate min samples must be positive.")
    if args.tva_terminal_weak_gate_min_samples <= 0:
        raise ValueError("weak tva terminal gate min samples must be positive.")
    if args.tva_terminal_gate_max_episodes <= 0 or args.tva_terminal_gate_max_starts_per_episode <= 0:
        raise ValueError("tva terminal gate audit sample sizes must be positive.")
    if args.max_selected_episodes is not None and args.max_selected_episodes <= 0:
        raise ValueError("max-selected-episodes must be positive when provided.")
    if args.max_selected_per_cycle_step is not None and args.max_selected_per_cycle_step <= 0:
        raise ValueError("max-selected-per-cycle-step must be positive when provided.")
    if args.postprocess_workers < 0:
        raise ValueError("postprocess-workers must be non-negative.")
    resume_seeds = {seed for seed, _ in args.initial_resume}
    unknown = sorted(resume_seeds.difference(args.seeds))
    if unknown:
        raise ValueError(f"initial resume provided for seeds not in --seeds: {unknown}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_root.mkdir(parents=True, exist_ok=True)

    rule_library = ensure_initial_library(args)
    prev_sd_checkpoint = args.initial_sd_checkpoint
    prev_tva_checkpoint = args.initial_tva_checkpoint
    if args.initial_tva_terminal_prior_policy != "auto":
        initial_tva_mode = args.initial_tva_terminal_prior_policy
    elif args.enable_tva_terminal_prior and not args.enable_tva_terminal_prior_gate:
        initial_tva_mode = "ungated"
    else:
        initial_tva_mode = "off"
    tva_terminal_prior_policy = make_tva_terminal_prior_policy(args, initial_tva_mode)
    tva_terminal_prior_gate_report: Path | None = None
    initial_resumes = dict(args.initial_resume)
    resumes: dict[int, Path | None] = {seed: initial_resumes.get(seed) for seed in args.seeds}

    write_json(
        args.output_root / "run_config.json",
        {
            "seeds": args.seeds,
            "gpu_ids": args.gpu_ids,
            "cycles": args.cycles,
            "cycle_steps": args.cycle_steps,
            "formal_eval_freq": args.formal_eval_freq,
            "formal_checkpoint_freq": args.formal_checkpoint_freq,
            "formal_eval_episodes": args.formal_eval_episodes,
            "exp_prefix": args.exp_prefix,
            "initial_rule_library": str(rule_library),
            "initial_sd_checkpoint": str(prev_sd_checkpoint) if prev_sd_checkpoint else None,
            "initial_tva_checkpoint": str(prev_tva_checkpoint) if prev_tva_checkpoint else None,
            "initial_tva_terminal_prior_policy": args.initial_tva_terminal_prior_policy,
            "initial_resumes": {str(k): str(v) for k, v in initial_resumes.items()},
            "enable_tva_audit": args.enable_tva_audit,
            "tva_epochs": args.tva_epochs,
            "tva_max_transitions": args.tva_max_transitions,
            "tva_gamma": args.tva_gamma,
            "tva_true_coef": args.tva_true_coef,
            "tva_shaped_coef": args.tva_shaped_coef,
            "tva_audit_max_records": args.tva_audit_max_records,
            "tva_audit_sample_mode": args.tva_audit_sample_mode,
            "enable_tva_shadow_update": args.enable_tva_shadow_update,
            "tva_shadow_family_top_k": args.tva_shadow_family_top_k,
            "tva_shadow_family_advantage_threshold": args.tva_shadow_family_advantage_threshold,
            "tva_shadow_min_hits": args.tva_shadow_min_hits,
            "tva_shadow_min_effect_hit_ratio": args.tva_shadow_min_effect_hit_ratio,
            "tva_shadow_min_positive_rate": args.tva_shadow_min_positive_rate,
            "tva_shadow_min_opposite_rate": args.tva_shadow_min_opposite_rate,
            "tva_shadow_min_directional_rate_margin": args.tva_shadow_min_directional_rate_margin,
            "tva_shadow_max_best_family_entropy": args.tva_shadow_max_best_family_entropy,
            "tva_shadow_target_opposite_margin": args.tva_shadow_target_opposite_margin,
            "tva_shadow_hit_ref": args.tva_shadow_hit_ref,
            "tva_shadow_positive_lr": args.tva_shadow_positive_lr,
            "tva_shadow_harmful_lr": args.tva_shadow_harmful_lr,
            "tva_shadow_advantage_scale": args.tva_shadow_advantage_scale,
            "enable_tva_consequence_gate": args.enable_tva_consequence_gate,
            "tva_consequence_horizon": args.tva_consequence_horizon,
            "tva_consequence_action_threshold": args.tva_consequence_action_threshold,
            "tva_consequence_min_outcome_margin": args.tva_consequence_min_outcome_margin,
            "tva_consequence_min_relation_count": args.tva_consequence_min_relation_count,
            "enable_tva_terminal_prior": args.enable_tva_terminal_prior,
            "tva_terminal_prior_weight": args.tva_terminal_prior_weight,
            "tva_terminal_prior_max_bonus": args.tva_terminal_prior_max_bonus,
            "tva_terminal_prior_normalize": args.tva_terminal_prior_normalize,
            "enable_tva_terminal_prior_weak_gate": args.enable_tva_terminal_prior_weak_gate,
            "tva_terminal_weak_prior_weight": args.tva_terminal_weak_prior_weight,
            "tva_terminal_weak_prior_max_bonus": args.tva_terminal_weak_prior_max_bonus,
            "enable_tva_terminal_prior_gate": args.enable_tva_terminal_prior_gate,
            "tva_terminal_gate_horizons": args.tva_terminal_gate_horizons,
            "tva_terminal_gate_max_episodes": args.tva_terminal_gate_max_episodes,
            "tva_terminal_gate_max_starts_per_episode": args.tva_terminal_gate_max_starts_per_episode,
            "tva_terminal_gate_min_samples": args.tva_terminal_gate_min_samples,
            "tva_terminal_gate_min_oracle_pearson": args.tva_terminal_gate_min_oracle_pearson,
            "tva_terminal_gate_min_sd_pearson": args.tva_terminal_gate_min_sd_pearson,
            "tva_terminal_gate_min_sd_r2": args.tva_terminal_gate_min_sd_r2,
            "tva_terminal_gate_max_sd_rmse_ratio": args.tva_terminal_gate_max_sd_rmse_ratio,
            "tva_terminal_gate_max_abs_delta_v": args.tva_terminal_gate_max_abs_delta_v,
            "tva_terminal_weak_gate_horizon_mode": args.tva_terminal_weak_gate_horizon_mode,
            "tva_terminal_weak_gate_min_samples": args.tva_terminal_weak_gate_min_samples,
            "tva_terminal_weak_gate_min_oracle_pearson": args.tva_terminal_weak_gate_min_oracle_pearson,
            "tva_terminal_weak_gate_min_sd_pearson": args.tva_terminal_weak_gate_min_sd_pearson,
            "tva_terminal_weak_gate_min_sd_r2": args.tva_terminal_weak_gate_min_sd_r2,
            "tva_terminal_weak_gate_max_sd_rmse_ratio": args.tva_terminal_weak_gate_max_sd_rmse_ratio,
            "rule_prior_allowed_statuses": args.rule_prior_allowed_statuses,
            "enable_rule_prior_stochastic_gate": args.enable_rule_prior_stochastic_gate,
            "rule_prior_min_apply_prob": args.rule_prior_min_apply_prob,
            "rule_prior_max_apply_prob": args.rule_prior_max_apply_prob,
            "rule_prior_confidence_prob_power": args.rule_prior_confidence_prob_power,
            "rule_prior_stochastic_eval": args.rule_prior_stochastic_eval,
            "rule_prior_stochastic_seed": args.rule_prior_stochastic_seed,
            "rule_prior_phase_match_mode": args.rule_prior_phase_match_mode,
            "enable_rule_prior_grounded_hop_phase": args.enable_rule_prior_grounded_hop_phase,
            "rule_required_status": args.rule_required_status,
            "max_online_rules_per_cycle": args.max_online_rules_per_cycle,
            "max_cycle_candidates": args.max_cycle_candidates,
            "rolling_min_cycles": args.rolling_min_cycles,
            "rolling_min_support": args.rolling_min_support,
            "rolling_min_advantage": args.rolling_min_advantage,
            "rolling_max_good_share": args.rolling_max_good_share,
            "rolling_weaken_max_good_share": args.rolling_weaken_max_good_share,
            "enable_v8_transition_rules": args.enable_v8_transition_rules,
            "enable_rule_effect_confidence": args.enable_rule_effect_confidence,
            "v8_transition_min_support": args.v8_transition_min_support,
            "v8_transition_min_seeds": args.v8_transition_min_seeds,
            "v8_max_false_intervention_rate": args.v8_max_false_intervention_rate,
            "v8_max_online_transition_rules_per_cycle": args.v8_max_online_transition_rules_per_cycle,
            "v8_max_cycle_transition_candidates": args.v8_max_cycle_transition_candidates,
            "confidence_default": args.confidence_default,
            "confidence_active_threshold": args.confidence_active_threshold,
            "confidence_harmful_threshold": args.confidence_harmful_threshold,
            "confidence_positive_context_threshold": args.confidence_positive_context_threshold,
            "confidence_hard_harmful_threshold": args.confidence_hard_harmful_threshold,
            "confidence_hard_harmful_min_hits": args.confidence_hard_harmful_min_hits,
            "confidence_conflict_keep_confidence_margin": args.confidence_conflict_keep_confidence_margin,
            "confidence_conflict_keep_harmful_margin": args.confidence_conflict_keep_harmful_margin,
            "protected_teacher_origins": args.protected_teacher_origins,
            "disable_teacher_core_conflict_protection": args.disable_teacher_core_conflict_protection,
            "selection_memory_policy": str(args.selection_memory_policy) if args.selection_memory_policy else None,
            "max_selected_episodes": args.max_selected_episodes,
            "max_selected_per_cycle_step": args.max_selected_per_cycle_step,
            "selection_preserve_sources": args.selection_preserve_sources,
            "postprocess_workers": args.postprocess_workers,
        },
    )

    for cycle in range(args.start_cycle, args.cycles + 1):
        write_cycle_report(
            args,
            cycle,
            rule_library,
            prev_sd_checkpoint,
            prev_tva_checkpoint,
            resumes,
            tva_terminal_prior_policy,
            tva_terminal_prior_gate_report,
        )
        resumes = run_seed_batch(
            args,
            cycle,
            args.seeds,
            rule_library,
            resumes,
            sd_checkpoint=prev_sd_checkpoint,
            tva_checkpoint=prev_tva_checkpoint,
            tva_terminal_prior_policy=tva_terminal_prior_policy,
        )
        export_replay_traces(args, cycle, resumes)

        shared_args = argparse.Namespace(**vars(args))
        shared_args.seed = args.selection_seed
        paths = cycle_paths(args.output_root, cycle)
        rule_library, prev_tva_checkpoint = summarize_cycle(
            shared_args,
            cycle,
            cycle_exp_name(args, cycle),
            paths,
            prev_sd_checkpoint,
            rule_library,
            prev_tva_checkpoint,
        )
        prev_sd_checkpoint = paths.sd_explainer
        tva_terminal_prior_policy, tva_terminal_prior_gate_report = run_tva_terminal_gate(
            args,
            cycle,
            paths.selection,
            prev_sd_checkpoint,
            prev_tva_checkpoint,
        )

    print("multi-seed shared-rule closed-loop run completed")
    print(f"final shared rule library: {rule_library}")
    print(f"final shared SD checkpoint: {prev_sd_checkpoint}")
    print(f"final shared TVA checkpoint: {prev_tva_checkpoint}")
    for seed in args.seeds:
        print(f"final TD-MPC2 seed {seed}: {resumes[seed]}")


if __name__ == "__main__":
    main()
