#!/usr/bin/env python3
"""Run v7 closed-loop Hopper training with periodic SD retrospective summaries."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PYTHON = "/home/anaconda3/envs/sdmcts_p100/bin/python"


@dataclass
class CyclePaths:
    root: Path
    manifest: Path
    selection: Path
    attributions: Path
    sd_explainer: Path
    sd_summary: Path
    tva_checkpoint: Path
    tva_summary: Path
    tva_counterfactual_audit: Path
    tva_shadow_update: Path
    tva_shadow_consequence_audit: Path
    tva_shadow_consequence_gated_update: Path
    counterfactuals: Path
    rules_v2: Path
    consistency_v3: Path
    reliable_v3: Path
    substate_v4: Path
    replay_v4: Path
    library_v4: Path
    substate_validation_v5: Path
    library_v5: Path
    recovery_next_library: Path
    success_fragments_v8: Path
    transition_actions_v8: Path
    transition_counterfactuals_v8: Path
    transition_rules_v8: Path
    transition_validation_v8: Path
    transition_safety_v8: Path
    transition_safe_rules_v8: Path
    rule_effect_audit: Path
    confidence_library: Path
    next_library: Path


def cycle_target_steps(args: argparse.Namespace, cycle: int) -> int:
    base = cycle * args.cycle_steps
    if args.align_resumed_eval and cycle > 1:
        return base + cycle - 1
    return base


def cycle_exp_name(args: argparse.Namespace, cycle: int) -> str:
    return f"{args.exp_prefix}_cycle{cycle:03d}_target{cycle_target_steps(args, cycle)}"


def run(cmd: list[str], env: dict[str, str] | None = None, dry_run: bool = False) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def cycle_paths(root: Path, cycle: int) -> CyclePaths:
    croot = root / f"cycle_{cycle:03d}"
    return CyclePaths(
        root=croot,
        manifest=croot / "manifests" / "manifest.json",
        selection=croot / "manifests" / "selected_trajectories.json",
        attributions=croot / "attributions" / "attributions.json",
        sd_explainer=croot / "sd_explainers" / f"sd_explainer_cycle_{cycle:03d}.pt",
        sd_summary=croot / "sd_explainers" / f"sd_explainer_cycle_{cycle:03d}_summary.json",
        tva_checkpoint=croot / "tva" / f"hopper_tva_cycle_{cycle:03d}.pt",
        tva_summary=croot / "tva" / f"hopper_tva_cycle_{cycle:03d}_summary.json",
        tva_counterfactual_audit=croot / "reports" / "tva_counterfactual_audit.json",
        tva_shadow_update=croot / "reports" / "tva_shadow_update.json",
        tva_shadow_consequence_audit=croot / "reports" / "tva_shadow_consequence_audit.json",
        tva_shadow_consequence_gated_update=croot / "reports" / "tva_shadow_consequence_gated_update.json",
        counterfactuals=croot / "attributions" / "counterfactuals_explainer.json",
        rules_v2=croot / "rules" / "state_action_rules_v2.json",
        consistency_v3=croot / "reports" / "rule_consistency_v3.json",
        reliable_v3=croot / "rules" / "reliable_rules_v3.json",
        substate_v4=croot / "rules" / "substate_rules_v4.json",
        replay_v4=croot / "reports" / "replay_rule_scoring_v4.json",
        library_v4=croot / "rules" / "library_v4.json",
        substate_validation_v5=croot / "reports" / "substate_rule_validation_v5.json",
        library_v5=croot / "rules" / "library_v5.json",
        recovery_next_library=croot / "rules" / f"library_closed_loop_cycle_{cycle:03d}_recovery.json",
        success_fragments_v8=croot / "reports" / "v8_success_fragments.json",
        transition_actions_v8=croot / "reports" / "v8_transition_actions.json",
        transition_counterfactuals_v8=croot / "reports" / "v8_transition_counterfactuals.json",
        transition_rules_v8=croot / "rules" / "v8_transition_rules.json",
        transition_validation_v8=croot / "reports" / "v8_transition_rule_validation.json",
        transition_safety_v8=croot / "reports" / "v8_transition_replay_safety.json",
        transition_safe_rules_v8=croot / "rules" / "v8_transition_replay_safe_rules.json",
        rule_effect_audit=croot / "reports" / "rule_effect_audit.json",
        confidence_library=croot / "rules" / f"library_closed_loop_cycle_{cycle:03d}_confidence.json",
        next_library=croot / "rules" / f"library_closed_loop_cycle_{cycle:03d}.json",
    )


def train_cycle(args: argparse.Namespace, cycle: int, rule_library: Path, resume_from: Path | None) -> Path:
    exp_name = cycle_exp_name(args, cycle)
    target_steps = cycle_target_steps(args, cycle)
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": args.python_bin,
            "WORLD_MODEL_KIND": "tdmpc2",
            "SD_ONLINE_SUPERVISION_ENABLED": "false",
            "STEPS": str(target_steps),
            "EVAL_FREQ": str(args.eval_freq),
            "CHECKPOINT_FREQ": str(args.checkpoint_freq),
            "EVAL_EPISODES": str(args.eval_episodes),
            "SEED_PRETRAIN_UPDATES": str(args.seed_pretrain_updates),
            "NUM_SAMPLES": str(args.num_samples),
            "NUM_ELITES": str(args.num_elites),
            "NUM_PI_TRAJS": str(args.num_pi_trajs),
            "ITERATIONS": str(args.iterations),
            "RULE_PRIOR_ENABLED": "true" if cycle > 1 or args.enable_rule_prior_cycle1 else "false",
            "RULE_PRIOR_MODE": args.rule_prior_mode,
            "RULE_PRIOR_OPPOSITE_PENALTY": str(args.rule_prior_opposite_penalty),
            "RULE_PRIOR_PREFERRED_CAP": str(args.rule_prior_preferred_cap),
            "RULE_PRIOR_CAP_PENALTY": str(args.rule_prior_cap_penalty),
            "RULE_PRIOR_ALLOWED_STATUSES": args.rule_prior_allowed_statuses,
            "RULE_LIBRARY": str(rule_library.resolve()),
        }
    )
    cmd = [
        "bash",
        str(REPO_ROOT / "scripts" / "run_hopper_v6_5k_rule_prior_smoke.sh"),
        str(args.gpu_id),
        exp_name,
        str(args.seed),
    ]
    if resume_from is not None:
        env["RESUME_FROM"] = str(resume_from.resolve())
    print(
        "\n".join(
            [
                f"[v7-cycle-{cycle:03d}] exp_name={exp_name}",
                f"[v7-cycle-{cycle:03d}] target_steps={target_steps}",
                f"[v7-cycle-{cycle:03d}] rule_library={rule_library}",
                f"[v7-cycle-{cycle:03d}] resume_from={resume_from if resume_from is not None else 'none'}",
                f"[v7-cycle-{cycle:03d}] rule_prior_enabled={env['RULE_PRIOR_ENABLED']}",
            ]
        ),
        flush=True,
    )
    run(cmd, env=env, dry_run=args.dry_run)
    return REPO_ROOT / "logs" / "hopper-hop" / str(args.seed) / exp_name / "models" / "final.full.pt"


def summarize_cycle(
    args: argparse.Namespace,
    cycle: int,
    exp_name: str,
    paths: CyclePaths,
    prev_sd_checkpoint: Path | None,
    base_rule_library: Path,
    prev_tva_checkpoint: Path | None = None,
) -> tuple[Path, Path | None]:
    py = args.python_bin
    common = [py]
    run(
        common
        + [
            str(SCRIPT_DIR / "build_hopper_manifest.py"),
            "--repo-root", str(REPO_ROOT),
            "--output", str(paths.manifest),
        ]
        + [
            item
            for window_cycle in range(max(1, cycle - args.summary_window_cycles + 1), cycle + 1)
            for item in (
                "--run-glob",
                cycle_exp_name(args, window_cycle),
            )
        ],
        dry_run=args.dry_run,
    )
    selection_cmd = common + [
            str(SCRIPT_DIR / "stratify_trajectories.py"),
            "--manifest", str(paths.manifest),
            "--output", str(paths.selection),
            "--good-quantile", str(args.good_quantile),
            "--bad-quantile", str(args.bad_quantile),
            "--random-per-cycle", str(args.random_per_cycle),
            "--score-mode", args.trajectory_score_mode,
            "--seed", str(args.seed + cycle),
    ]
    if getattr(args, "selection_memory_policy", None) is not None:
        selection_cmd.extend(["--memory-policy", str(args.selection_memory_policy)])
    if getattr(args, "max_selected_episodes", None) is not None:
        selection_cmd.extend(["--max-selected-episodes", str(args.max_selected_episodes)])
    if getattr(args, "max_selected_per_cycle_step", None) is not None:
        selection_cmd.extend(["--max-selected-per-cycle-step", str(args.max_selected_per_cycle_step)])
    for source in getattr(args, "selection_preserve_sources", []) or []:
        selection_cmd.extend(["--preserve-source", str(source)])
    run(selection_cmd, dry_run=args.dry_run)
    run(
        common + [
            str(SCRIPT_DIR / "attribute_failures.py"),
            "--selection", str(paths.selection),
            "--output", str(paths.attributions),
        ],
        dry_run=args.dry_run,
    )
    fit_cmd = common + [
        str(SCRIPT_DIR / "fit_sd_explainer.py"),
        "--selection", str(paths.selection),
        "--output", str(paths.sd_explainer),
        "--summary", str(paths.sd_summary),
        "--seed", str(args.seed + cycle),
        "--epochs", str(args.sd_epochs),
        "--max-transitions", str(args.sd_max_transitions),
        "--batch-size", str(args.sd_batch_size),
        "--lr", str(args.sd_lr),
        "--device", args.sd_device,
        "--use-transition-residual",
        "--transition-residual-scale", str(args.sd_transition_residual_scale),
    ]
    if prev_sd_checkpoint is not None:
        fit_cmd.extend(["--init-checkpoint", str(prev_sd_checkpoint)])
    run(fit_cmd, dry_run=args.dry_run)
    tva_checkpoint = prev_tva_checkpoint
    if getattr(args, "enable_tva_audit", False):
        tva_cmd = common + [
            str(SCRIPT_DIR / "train_hopper_tva_explainer.py"),
            "--selection", str(paths.selection),
            "--sd-checkpoint", str(paths.sd_explainer),
            "--output", str(paths.tva_checkpoint),
            "--summary", str(paths.tva_summary),
            "--seed", str(args.seed + cycle),
            "--epochs", str(args.tva_epochs),
            "--max-transitions", str(args.tva_max_transitions),
            "--batch-size", str(args.tva_batch_size),
            "--lr", str(args.tva_lr),
            "--gamma", str(args.tva_gamma),
            "--true-coef", str(args.tva_true_coef),
            "--shaped-coef", str(args.tva_shaped_coef),
            "--hidden", str(args.tva_hidden),
            "--device", args.tva_device,
        ]
        if prev_tva_checkpoint is not None:
            tva_cmd.extend(["--init-checkpoint", str(prev_tva_checkpoint)])
        run(tva_cmd, dry_run=args.dry_run)
        tva_checkpoint = paths.tva_checkpoint
    run(
        common + [
            str(SCRIPT_DIR / "counterfactual_actions.py"),
            "--attributions", str(paths.attributions),
            "--output", str(paths.counterfactuals),
            "--model-checkpoint", str(paths.sd_explainer),
            "--max-states-per-episode", str(args.max_states_per_episode),
            "--horizon", str(args.counterfactual_horizon),
            "--workers", str(args.postprocess_workers),
        ],
        dry_run=args.dry_run,
    )
    if getattr(args, "enable_tva_audit", False):
        run(
            common + [
                str(SCRIPT_DIR / "audit_hopper_tva_counterfactuals.py"),
                "--counterfactuals", str(paths.counterfactuals),
                "--sd-checkpoint", str(paths.sd_explainer),
                "--tva-checkpoint", str(paths.tva_checkpoint),
                "--output", str(paths.tva_counterfactual_audit),
                "--horizon", str(args.counterfactual_horizon),
                "--max-records", str(args.tva_audit_max_records),
                "--sample-mode", args.tva_audit_sample_mode,
                "--device", args.tva_device,
            ],
            dry_run=args.dry_run,
        )
    run(common + [str(SCRIPT_DIR / "induce_state_action_rules_v2.py"), "--counterfactuals", str(paths.counterfactuals), "--output", str(paths.rules_v2)], dry_run=args.dry_run)
    run(common + [str(SCRIPT_DIR / "analyze_rule_consistency.py"), "--counterfactuals", str(paths.counterfactuals), "--rules", str(paths.rules_v2), "--output", str(paths.consistency_v3), "--reliable-output", str(paths.reliable_v3)], dry_run=args.dry_run)
    run(common + [str(SCRIPT_DIR / "analyze_substate_rules.py"), "--counterfactuals", str(paths.counterfactuals), "--output", str(paths.substate_v4)], dry_run=args.dry_run)
    run(common + [str(SCRIPT_DIR / "replay_rule_scoring.py"), "--attributions", str(paths.attributions), "--rules", str(paths.reliable_v3), "--output", str(paths.replay_v4), "--workers", str(args.postprocess_workers)], dry_run=args.dry_run)
    run(common + [str(SCRIPT_DIR / "build_rule_library.py"), "--reliable-rules", str(paths.reliable_v3), "--substate-rules", str(paths.substate_v4), "--output", str(paths.library_v4)], dry_run=args.dry_run)
    run(common + [str(SCRIPT_DIR / "validate_substate_rules.py"), "--attributions", str(paths.attributions), "--rules", str(paths.substate_v4), "--output", str(paths.substate_validation_v5), "--workers", str(args.postprocess_workers)], dry_run=args.dry_run)
    run(common + [str(SCRIPT_DIR / "update_rule_library_status.py"), "--library", str(paths.library_v4), "--replay", str(paths.replay_v4), "--substate-validation", str(paths.substate_validation_v5), "--output", str(paths.library_v5)], dry_run=args.dry_run)
    run(
        common + [
            str(SCRIPT_DIR / "closed_loop_rule_update.py"),
            "--base-library", str(base_rule_library),
            "--cycle-library", str(paths.library_v5),
            "--output", str(paths.recovery_next_library if args.enable_v8_transition_rules else paths.next_library),
            "--cycle", str(cycle),
            "--required-status", args.rule_required_status,
            "--max-online-rules-per-cycle", str(args.max_online_rules_per_cycle),
            "--max-cycle-candidates", str(args.max_cycle_candidates),
            "--rolling-min-cycles", str(args.rolling_min_cycles),
            "--rolling-min-support", str(args.rolling_min_support),
            "--rolling-min-advantage", str(args.rolling_min_advantage),
            "--rolling-max-good-share", str(args.rolling_max_good_share),
            "--rolling-weaken-max-good-share", str(args.rolling_weaken_max_good_share),
        ],
        dry_run=args.dry_run,
    )
    if not args.enable_v8_transition_rules:
        return finalize_cycle_library(
            args,
            paths,
            apply_rule_effect_confidence(args, paths, paths.next_library, []),
        ), tva_checkpoint

    trace_dir_args: list[str] = []
    seeds = getattr(args, "seeds", [getattr(args, "seed", 1)])
    for seed in seeds:
        run_dir = REPO_ROOT / "logs" / "hopper-hop" / str(seed) / exp_name / "retrospective_train_traces"
        step_dirs = sorted(run_dir.glob("step_*/episode_csv"))
        if step_dirs:
            trace_dir_args.extend(["--trace-dir", str(step_dirs[-1])])
        elif args.dry_run:
            step_dir = run_dir / f"step_{cycle_target_steps(args, cycle) + 1:07d}" / "episode_csv"
            trace_dir_args.extend(["--trace-dir", str(step_dir)])

    if not trace_dir_args:
        # No replay traces were exported; keep the recovery-rule library as the
        # cycle output so the caller can continue without a special case.
        run(
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    f"Path({str(paths.next_library)!r}).write_text(Path({str(paths.recovery_next_library)!r}).read_text())"
                ),
            ],
            dry_run=args.dry_run,
        )
        return finalize_cycle_library(
            args,
            paths,
            apply_rule_effect_confidence(args, paths, paths.next_library, []),
        ), tva_checkpoint

    run(
        common + [
            str(SCRIPT_DIR / "mine_hopper_v8_success_segments.py"),
            *trace_dir_args,
            "--rule-library", str(paths.recovery_next_library),
            "--output", str(paths.success_fragments_v8),
            "--min-length", str(args.v8_success_min_length),
            "--lookahead", str(args.v8_success_lookahead),
            "--workers", str(args.postprocess_workers),
        ],
        dry_run=args.dry_run,
    )
    run(
        common + [
            str(SCRIPT_DIR / "mine_hopper_v8_transition_actions.py"),
            "--fragments", str(paths.success_fragments_v8),
            "--output", str(paths.transition_actions_v8),
            "--min-support", str(args.v8_transition_min_support),
            "--min-seeds", str(args.v8_transition_min_seeds),
        ],
        dry_run=args.dry_run,
    )
    run(
        common + [
            str(SCRIPT_DIR / "counterfactual_hopper_v8_transition_actions.py"),
            "--evidence", str(paths.transition_actions_v8),
            "--output", str(paths.transition_counterfactuals_v8),
            "--model-checkpoint", str(paths.sd_explainer),
            "--max-records-per-key", str(args.v8_counterfactual_max_records_per_key),
            "--min-support", str(args.v8_counterfactual_min_support),
            "--min-advantage", str(args.v8_counterfactual_min_advantage),
        ],
        dry_run=args.dry_run,
    )
    run(
        common + [
            str(SCRIPT_DIR / "induce_hopper_v8_transition_rules.py"),
            "--observational", str(paths.transition_actions_v8),
            "--counterfactuals", str(paths.transition_counterfactuals_v8),
            "--output", str(paths.transition_rules_v8),
            "--min-seeds", str(args.v8_transition_min_seeds),
        ],
        dry_run=args.dry_run,
    )
    run(
        common + [
            str(SCRIPT_DIR / "validate_hopper_v8_transition_rules.py"),
            "--rules", str(paths.transition_rules_v8),
            "--observational", str(paths.transition_actions_v8),
            "--counterfactuals", str(paths.transition_counterfactuals_v8),
            "--output", str(paths.transition_validation_v8),
            "--min-support", str(args.v8_validation_min_support),
            "--min-seeds", str(args.v8_validation_min_seeds),
            "--min-advantage", str(args.v8_validation_min_advantage),
        ],
        dry_run=args.dry_run,
    )
    train_replay_args = [
        item
        for seed in seeds
        for item in (
            "--train-replay-export",
            str(paths.root / "train_replay_exports" / f"seed{seed}.json"),
        )
        if (paths.root / "train_replay_exports" / f"seed{seed}.json").exists() or args.dry_run
    ]
    run(
        common + [
            str(SCRIPT_DIR / "audit_hopper_v8_transition_replay_safety.py"),
            "--validation", str(paths.transition_validation_v8),
            *train_replay_args,
            "--output", str(paths.transition_safety_v8),
            "--max-episodes-per-seed", str(args.v8_safety_max_episodes_per_seed),
        ],
        dry_run=args.dry_run,
    )
    run(
        common + [
            str(SCRIPT_DIR / "export_hopper_v8_replay_safe_rules.py"),
            "--safety", str(paths.transition_safety_v8),
            "--output", str(paths.transition_safe_rules_v8),
            "--max-false-intervention-rate", str(args.v8_max_false_intervention_rate),
        ],
        dry_run=args.dry_run,
    )
    run(
        common + [
            str(SCRIPT_DIR / "closed_loop_rule_update.py"),
            "--base-library", str(paths.recovery_next_library),
            "--cycle-library", str(paths.transition_safe_rules_v8),
            "--output", str(paths.next_library),
            "--cycle", str(cycle),
            "--required-status", "v8_replay_safe_candidate",
            "--allow-source", "v8_transition_replay_safety",
            "--max-online-rules-per-cycle", str(args.v8_max_online_transition_rules_per_cycle),
            "--max-cycle-candidates", str(args.v8_max_cycle_transition_candidates),
            "--rolling-min-cycles", str(args.rolling_min_cycles),
            "--rolling-min-support", str(args.rolling_min_support),
            "--rolling-min-advantage", str(args.rolling_min_advantage),
            "--rolling-max-good-share", str(args.rolling_max_good_share),
            "--rolling-weaken-max-good-share", str(args.rolling_weaken_max_good_share),
        ],
        dry_run=args.dry_run,
    )
    return finalize_cycle_library(
        args,
        paths,
        apply_rule_effect_confidence(args, paths, paths.next_library, trace_dir_args),
    ), tva_checkpoint


def finalize_cycle_library(args: argparse.Namespace, paths: CyclePaths, rule_library: Path) -> Path:
    if not getattr(args, "enable_tva_shadow_update", False):
        if getattr(args, "enable_tva_consequence_gate", False):
            print("[tva-consequence] skipped: --enable-tva-shadow-update is false", flush=True)
        return rule_library
    if not getattr(args, "enable_tva_audit", False):
        print("[tva-shadow] skipped: --enable-tva-audit is false", flush=True)
        return rule_library

    cmd = [
        args.python_bin,
        str(SCRIPT_DIR / "apply_tva_shadow_update.py"),
        "--library", str(rule_library),
        "--tva-counterfactual-audit", str(paths.tva_counterfactual_audit),
        "--output", str(paths.tva_shadow_update),
        "--continue-score-min", str(args.rule_effect_continue_score_min),
        "--family-top-k", str(args.tva_shadow_family_top_k),
        "--family-advantage-threshold", str(args.tva_shadow_family_advantage_threshold),
        "--min-effect-hit-ratio", str(args.tva_shadow_min_effect_hit_ratio),
        "--min-positive-rate", str(args.tva_shadow_min_positive_rate),
        "--min-opposite-rate", str(args.tva_shadow_min_opposite_rate),
        "--min-directional-rate-margin", str(args.tva_shadow_min_directional_rate_margin),
        "--max-best-family-entropy", str(args.tva_shadow_max_best_family_entropy),
        "--target-opposite-margin", str(args.tva_shadow_target_opposite_margin),
        "--default-confidence", str(args.confidence_default),
        "--active-threshold", str(args.confidence_active_threshold),
        "--retired-threshold", str(args.confidence_retired_threshold),
        "--min-hits", str(args.tva_shadow_min_hits),
        "--hit-ref", str(args.tva_shadow_hit_ref),
        "--positive-lr", str(args.tva_shadow_positive_lr),
        "--harmful-lr", str(args.tva_shadow_harmful_lr),
        "--advantage-scale", str(args.tva_shadow_advantage_scale),
    ]
    if paths.rule_effect_audit.exists() or args.dry_run:
        cmd.extend(["--rule-effect-audit", str(paths.rule_effect_audit)])
    run(cmd, dry_run=args.dry_run)
    if getattr(args, "enable_tva_consequence_gate", False):
        run(
            [
                args.python_bin,
                str(SCRIPT_DIR / "audit_tva_shadow_consequences.py"),
                "--shadow-update", str(paths.tva_shadow_update),
                "--tva-counterfactual-audit", str(paths.tva_counterfactual_audit),
                "--output", str(paths.tva_shadow_consequence_audit),
                "--horizon", str(args.tva_consequence_horizon),
                "--action-threshold", str(args.tva_consequence_action_threshold),
                "--continue-score-min", str(args.rule_effect_continue_score_min),
            ],
            dry_run=args.dry_run,
        )
        run(
            [
                args.python_bin,
                str(SCRIPT_DIR / "apply_tva_consequence_gate.py"),
                "--shadow-update", str(paths.tva_shadow_update),
                "--consequence-audit", str(paths.tva_shadow_consequence_audit),
                "--output", str(paths.tva_shadow_consequence_gated_update),
                "--min-outcome-margin", str(args.tva_consequence_min_outcome_margin),
                "--min-relation-count", str(args.tva_consequence_min_relation_count),
                "--active-threshold", str(args.confidence_active_threshold),
                "--retired-threshold", str(args.confidence_retired_threshold),
            ],
            dry_run=args.dry_run,
        )
    return rule_library


def apply_rule_effect_confidence(
    args: argparse.Namespace,
    paths: CyclePaths,
    rule_library: Path,
    trace_dir_args: list[str],
) -> Path:
    """Optionally convert the cycle library into confidence-weighted rules."""

    if not getattr(args, "enable_rule_effect_confidence", False):
        return rule_library
    if not trace_dir_args:
        return rule_library

    py = args.python_bin
    common = [py]
    run(
        common + [
            str(SCRIPT_DIR / "audit_hopper_v8_widefunnel_rule_effects.py"),
            "--rule-library", str(rule_library),
            "--output", str(paths.rule_effect_audit),
            "--high-episode-reward", str(args.rule_effect_high_episode_reward),
            "--continue-score-min", str(args.rule_effect_continue_score_min),
            "--action-threshold", str(args.rule_effect_action_threshold),
            "--audit-status", "cycle_candidate",
            "--audit-status", "cycle_supported",
            "--audit-status", "rolling_validated",
            "--audit-status", "v8_replay_safe_candidate",
            "--audit-status", "confidence_active",
            "--audit-status", "confidence_observing",
            "--workers", str(args.postprocess_workers),
            *trace_dir_args,
        ],
        dry_run=args.dry_run,
    )
    run(
        common + [
            str(SCRIPT_DIR / "apply_rule_effect_confidence.py"),
            "--library", str(rule_library),
            "--audit", str(paths.rule_effect_audit),
            "--output", str(paths.confidence_library),
            "--target-status", "cycle_candidate",
            "--target-status", "cycle_supported",
            "--target-status", "rolling_validated",
            "--target-status", "v8_replay_safe_candidate",
            "--target-status", "confidence_active",
            "--target-status", "confidence_observing",
            "--target-status", "confidence_retired",
            "--default-confidence", str(args.confidence_default),
            "--active-threshold", str(args.confidence_active_threshold),
            "--retired-threshold", str(args.confidence_retired_threshold),
            "--min-effect-hits", str(args.confidence_min_effect_hits),
            "--hit-ref", str(args.confidence_hit_ref),
            "--positive-threshold", str(args.confidence_positive_threshold),
            "--harmful-threshold", str(args.confidence_harmful_threshold),
            "--positive-lr", str(args.confidence_positive_lr),
            "--harmful-lr", str(args.confidence_harmful_lr),
            "--absence-decay", str(args.confidence_absence_decay),
            "--low-hit-decay", str(args.confidence_low_hit_decay),
            "--context-floor", str(args.confidence_context_floor),
            "--positive-context-threshold", str(args.confidence_positive_context_threshold),
            "--hard-harmful-threshold", str(args.confidence_hard_harmful_threshold),
            "--hard-harmful-min-hits", str(args.confidence_hard_harmful_min_hits),
            "--hard-harmful-margin", str(args.confidence_hard_harmful_margin),
            "--conflict-keep-confidence-margin", str(args.confidence_conflict_keep_confidence_margin),
            "--conflict-keep-harmful-margin", str(args.confidence_conflict_keep_harmful_margin),
            "--conflict-demote-margin", str(args.confidence_conflict_demote_margin),
            "--protected-teacher-origins",
            str(getattr(args, "protected_teacher_origins", "teacher_core_standard_hop")),
            *(
                ["--no-protect-teacher-conflicts"]
                if getattr(args, "disable_teacher_core_conflict_protection", False)
                else []
            ),
        ],
        dry_run=args.dry_run,
    )
    return paths.confidence_library


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--start-cycle", type=int, default=1)
    parser.add_argument("--cycle-steps", type=int, default=20000)
    parser.add_argument("--align-resumed-eval", action="store_true")
    parser.add_argument("--exp-prefix", default="tdmpc2_v7_closed_loop")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "results" / "sd_retrospective_rule_induction" / "closed_loop" / "hopper_v7")
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON)
    parser.add_argument("--initial-rule-library", type=Path, default=REPO_ROOT / "results" / "sd_retrospective_rule_induction" / "rules" / "library_v5.json")
    parser.add_argument("--initial-sd-checkpoint", type=Path, default=None)
    parser.add_argument("--initial-tva-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enable-rule-prior-cycle1", action="store_true")
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--checkpoint-freq", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--seed-pretrain-updates", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--num-elites", type=int, default=8)
    parser.add_argument("--num-pi-trajs", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--rule-prior-mode", default="penalty_only")
    parser.add_argument("--rule-prior-opposite-penalty", type=float, default=0.02)
    parser.add_argument("--rule-prior-preferred-cap", type=float, default=0.70)
    parser.add_argument("--rule-prior-cap-penalty", type=float, default=0.02)
    parser.add_argument("--rule-prior-allowed-statuses", default="replay_validated")
    parser.add_argument("--rule-required-status", default="replay_validated")
    parser.add_argument("--summary-window-cycles", type=int, default=1)
    parser.add_argument("--trajectory-score-mode", choices=["reward", "stage", "hybrid"], default="reward")
    parser.add_argument("--max-online-rules-per-cycle", type=int, default=2)
    parser.add_argument("--max-cycle-candidates", type=int, default=12)
    parser.add_argument("--rolling-min-cycles", type=int, default=2)
    parser.add_argument("--rolling-min-support", type=int, default=6)
    parser.add_argument("--rolling-min-advantage", type=float, default=0.05)
    parser.add_argument("--rolling-max-good-share", type=float, default=0.25)
    parser.add_argument("--rolling-weaken-max-good-share", type=float, default=0.25)
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
        "--disable-teacher-core-conflict-protection",
        action="store_true",
        default=False,
        help="Disable the post-2026-05-25 teacher-core conflict protection to reproduce the pre-protection update semantics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cycle_steps <= 0 or args.cycles <= 0 or args.start_cycle <= 0:
        raise ValueError("cycles, start_cycle, and cycle_steps must be positive.")
    if args.start_cycle > args.cycles:
        raise ValueError("start_cycle must be <= cycles.")
    if args.max_selected_episodes is not None and args.max_selected_episodes <= 0:
        raise ValueError("max-selected-episodes must be positive when provided.")
    if args.max_selected_per_cycle_step is not None and args.max_selected_per_cycle_step <= 0:
        raise ValueError("max-selected-per-cycle-step must be positive when provided.")
    if args.postprocess_workers < 0:
        raise ValueError("postprocess-workers must be non-negative.")

    rule_library = args.initial_rule_library
    prev_sd_checkpoint = args.initial_sd_checkpoint
    prev_tva_checkpoint = args.initial_tva_checkpoint
    resume_from = args.resume_from
    for cycle in range(args.start_cycle, args.cycles + 1):
        exp_name = cycle_exp_name(args, cycle)
        resume_from = train_cycle(args, cycle, rule_library, resume_from)
        paths = cycle_paths(args.output_root, cycle)
        rule_library, prev_tva_checkpoint = summarize_cycle(
            args,
            cycle,
            exp_name,
            paths,
            prev_sd_checkpoint,
            rule_library,
            prev_tva_checkpoint,
        )
        prev_sd_checkpoint = paths.sd_explainer

    print("closed-loop run configured successfully")
    print(f"final rule library: {rule_library}")
    print(f"final SD checkpoint: {prev_sd_checkpoint}")
    print(f"final TVA checkpoint: {prev_tva_checkpoint}")
    print(f"final TD-MPC2 full-state checkpoint: {resume_from}")


if __name__ == "__main__":
    main()
