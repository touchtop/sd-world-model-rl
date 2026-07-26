#!/usr/bin/env python3
"""Build a read-only manifest for Hopper retrospective analysis.

The script indexes existing Hopper run directories and writes a manifest under
results/sd_retrospective_rule_induction/. It never writes into logs/hopper-hop
or TD-MPC2 source directories.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_RUN_GLOBS = (
    "sd_hopper_100k_pureenv_sdinit_mech_sd_budget_seed*_20260501_run1",
    "sd_hopper_200k_pureenv_sdinit_mech_sd_budget_seed*_resume100k_20260502_run1",
    "sd_hopper_300k_pureenv_sdinit_mech_sd_official_budget_seed*_resume200k_20260504_run1",
)


def repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def read_eval_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "step": int(float(row["step"])),
                    "episode_reward": float(row["episode_reward"]),
                }
            )
    return rows


def discover_official_eval_summaries(run_dir: Path, repo_root: Path) -> list[dict[str, str]]:
    official_root = run_dir / "official_eval"
    if not official_root.exists():
        return []
    summaries: list[dict[str, str]] = []
    for summary in sorted(official_root.glob("step_*/eval_episode_summary.csv")):
        summaries.append(
            {
                "step_dir": summary.parent.name,
                "summary_csv": repo_relative(summary, repo_root),
            }
        )
    return summaries


def discover_train_replay_summaries(run_dir: Path, repo_root: Path) -> list[dict[str, str]]:
    replay_root = run_dir / "retrospective_train_traces"
    if not replay_root.exists():
        return []
    summaries: list[dict[str, str]] = []
    for summary in sorted(replay_root.glob("step_*/eval_episode_summary.csv")):
        summaries.append(
            {
                "step_dir": summary.parent.name,
                "summary_csv": repo_relative(summary, repo_root),
                "source": "train_replay",
            }
        )
    return summaries


def discover_runs(repo_root: Path, log_root: Path, run_globs: tuple[str, ...]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for seed_dir in sorted(log_root.iterdir()):
        if not seed_dir.is_dir():
            continue
        try:
            seed: int | str = int(seed_dir.name)
        except ValueError:
            seed = seed_dir.name
        for pattern in run_globs:
            for run_dir in sorted(seed_dir.glob(pattern)):
                eval_csv = run_dir / "eval.csv"
                train_csv = run_dir / "train.csv"
                eval_full_csv = run_dir / "eval_full.csv"
                runs.append(
                    {
                        "seed": seed,
                        "run_name": run_dir.name,
                        "run_dir": repo_relative(run_dir, repo_root),
                        "eval_csv": repo_relative(eval_csv, repo_root) if eval_csv.exists() else None,
                        "eval_full_csv": repo_relative(eval_full_csv, repo_root)
                        if eval_full_csv.exists()
                        else None,
                        "train_csv": repo_relative(train_csv, repo_root) if train_csv.exists() else None,
                        "eval_rows": read_eval_csv(eval_csv),
                        "official_eval_summaries": discover_official_eval_summaries(run_dir, repo_root),
                        "train_replay_summaries": discover_train_replay_summaries(run_dir, repo_root),
                    }
                )
    return runs


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    repo_root_default = script_path.parents[2]
    output_default = (
        repo_root_default
        / "results"
        / "sd_retrospective_rule_induction"
        / "manifests"
        / "hopper_offline_v1_manifest.json"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root_default)
    parser.add_argument("--log-root", type=Path, default=repo_root_default / "logs" / "hopper-hop")
    parser.add_argument("--output", type=Path, default=output_default)
    parser.add_argument("--run-glob", action="append", dest="run_globs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    log_root = args.log_root.resolve()
    run_globs = tuple(args.run_globs) if args.run_globs else DEFAULT_RUN_GLOBS

    if not log_root.exists():
        raise FileNotFoundError(f"log root does not exist: {log_root}")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "log_root": repo_relative(log_root, repo_root),
        "run_globs": list(run_globs),
        "write_policy": "results_only",
        "runs": discover_runs(repo_root, log_root, run_globs),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"wrote manifest: {args.output}")
    print(f"indexed runs: {len(manifest['runs'])}")


if __name__ == "__main__":
    main()
