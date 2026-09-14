"""Aggregate §8 job records: paired seed-level bootstrap and per-topology tables.

Run: python -m experiments.beer_game_graph.summarize --dir results/DIR [--ref sd_oracle]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BOOT = 10_000
BOOT_SEED = 20260913


def paired_bootstrap(a: np.ndarray, b: np.ndarray, rng) -> dict:
    d = a - b
    idx = rng.integers(0, len(d), size=(BOOT, len(d)))
    means = d[idx].mean(1)
    return {"mean_diff": float(d.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "n": int(len(d)), "all_positive": bool((d > 0).all())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--baseline", default="bb")
    args = ap.parse_args()
    root = Path(args.dir)
    recs = [json.loads(p.read_text()) for p in root.glob("*.json") if p.name != "summary.json"]
    by = defaultdict(dict)  # (topology, budget, arm, leaf) -> seed -> record
    for r in recs:
        by[(r["topology"], r["budget"], r["arm"], r["leaf"])][r["seed"]] = r
    rng = np.random.default_rng(BOOT_SEED)
    table, pairs = {}, {}
    for key, seeds in sorted(by.items()):
        ret = np.array([seeds[s]["eval"]["mean_return"] for s in sorted(seeds)])
        row = {"seeds": sorted(seeds), "mean_return": float(ret.mean()), "std": float(ret.std(ddof=1)) if len(ret) > 1 else 0.0,
               "worst_seed": float(ret.min())}
        first = next(iter(seeds.values()))
        for k in ("rollout_rmse", "imagined_violation", "model_calls"):
            vals = [seeds[s]["eval"].get(k) for s in sorted(seeds) if seeds[s]["eval"].get(k) is not None]
            if vals:
                row[k] = ({h: float(np.mean([v[h] for v in vals])) for h in vals[0]} if k == "rollout_rmse"
                          else float(np.mean([v["rate"] if isinstance(v, dict) else v for v in vals])))
        if "theta_final" in first:
            row["theta_dim"] = first.get("theta_dim")
            for p in ("demand_high", "demand_step_time"):
                errs = [np.abs(np.asarray(seeds[s]["theta_final"][p]) - (8.0 if p == "demand_high" else 5.0)).mean()
                        for s in sorted(seeds)]
                row[f"{p}_abs_err"] = float(np.mean(errs))
        row["model_params"] = first.get("model_params")
        table["|".join(map(str, key))] = row
    for (t, b, arm, leaf), seeds in by.items():
        base = by.get((t, b, args.baseline, leaf))
        if arm == args.baseline or not base:
            continue
        common = sorted(set(seeds) & set(base))
        if len(common) < 2:
            continue
        a = np.array([seeds[s]["eval"]["mean_return"] for s in common])
        c = np.array([base[s]["eval"]["mean_return"] for s in common])
        pairs[f"{t}|{b}|{arm}-{args.baseline}|{leaf}"] = paired_bootstrap(a, c, rng)
    summary = {"n_records": len(recs), "bootstrap": {"resamples": BOOT, "seed": BOOT_SEED}, "table": table, "paired_vs_" + args.baseline: pairs}
    (root / "summary.json").write_text(json.dumps(summary, indent=1))
    for k, v in table.items():
        print(f"{k:32s} mean {v['mean_return']:9.1f}  std {v['std']:7.1f}  worst {v['worst_seed']:9.1f}  n={len(v['seeds'])}")
    for k, v in pairs.items():
        print(f"{k:32s} diff {v['mean_diff']:+8.1f}  CI {v['ci95'][0]:+8.1f} {v['ci95'][1]:+8.1f}  all_pos={v['all_positive']}")


if __name__ == "__main__":
    main()
