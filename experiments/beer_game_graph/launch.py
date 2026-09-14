"""Job-matrix launcher for the §8 protocol (CPU process pool, resumable).

Run: python -m experiments.beer_game_graph.launch --out results/DIR --topologies T1 --arms sd bb sd_oracle basestock \
        --budgets 1000 --seeds 0 1 2 3 4 --workers 4
Existing result files are skipped by ``run_arm``; rerun the same command to resume.
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.beer_game_graph.common import ARMS, job_name


def run(cmd, log_path):
    with open(log_path, "a") as f:
        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT,
                               env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                               preexec_fn=lambda: os.nice(10))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topologies", nargs="+", default=["T1", "T2", "T3", "T4"])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--budgets", nargs="+", type=int, default=[1000, 3000, 10000])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--leaf", default="rollout")
    ap.add_argument("--simulations", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4,
                    help="maximum concurrent subprocesses (kept low by default for shared servers)")
    ap.add_argument("--resume-checkpoints", action="store_true",
                    help="evaluate existing .pt checkpoints instead of retraining jobs without JSON")
    args = ap.parse_args()
    out = Path(args.out); (out / "logs").mkdir(parents=True, exist_ok=True)
    jobs = []
    for t, arm, b, s in itertools.product(args.topologies, args.arms, args.budgets, args.seeds):
        if arm == "basestock" and (b != args.budgets[0]):
            continue  # no training; one record per seed
        if arm == "sd_node" and t in ("T1", "T2"):
            continue  # single retail node: identical to sd
        name = job_name(t, arm, b, s, args.leaf)
        if (out / f"{name}.json").exists():
            continue
        cmd = [sys.executable, "-m", "experiments.beer_game_graph.run_arm", "--topology", t, "--arm", arm,
               "--budget", str(b), "--seed", str(s), "--out", str(out), "--leaf", args.leaf,
               "--simulations", str(args.simulations)]
        if args.resume_checkpoints:
            cmd.append("--resume-checkpoints")
        jobs.append((cmd, out / "logs" / f"{name}.log"))
    print(f"{len(jobs)} jobs, {args.workers} workers", flush=True)
    # Smallest budget first (per the wave plan); sd_mpc depends on the sd checkpoint so it runs in a second wave.
    jobs.sort(key=lambda j: ("sd_mpc" in j[0], int(j[0][j[0].index("--budget") + 1])))
    first = [j for j in jobs if "sd_mpc" not in j[0]]
    second = [j for j in jobs if "sd_mpc" in j[0]]
    codes = []
    for wave in (first, second):
        with ThreadPoolExecutor(args.workers) as pool:
            codes += list(pool.map(lambda j: run(*j), wave))
    failed = [j[1].stem for j, c in zip(first + second, codes) if c != 0]
    print(f"done; failed: {failed}")


if __name__ == "__main__":
    main()
