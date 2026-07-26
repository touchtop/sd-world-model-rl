#!/usr/bin/env python3
"""Evaluate the Sokoban SD world model on Boxoban levels.

The experiment compares an unguided push-level uniform-cost baseline with an
SD-guided best-first search that uses reachability, deadlock checks, and
box-goal matching lower bounds.  It also checks tensor-adapter transition
consistency for legal macro pushes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sokoban import SokobanGoalDirectedSDModel, best_first_search, load_boxoban_levels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        default="datasets/sokoban_boxoban/boxoban-levels/medium/train/000.txt",
        help="Boxoban file or directory to evaluate.",
    )
    parser.add_argument("--max-levels", type=int, default=20)
    parser.add_argument("--max-expansions", type=int, default=2000)
    parser.add_argument("--heuristic-weight", type=float, default=1.0)
    parser.add_argument(
        "--out",
        default="results/sokoban_sd_world_model/eval_summary.json",
        help="JSON output path.",
    )
    return parser


def _transition_consistency(model: SokobanGoalDirectedSDModel) -> Dict[str, int]:
    total = 0
    ok = 0
    state = model.initial_state
    for action in model.legal_pushes(state):
        total += 1
        direct = model.step_state(state, action)
        tensor_state = model.encode_tensor_state(state).unsqueeze(0)
        tensor_action = torch.tensor([action.flat_index], dtype=torch.long)
        tensor_next = model.forward(tensor_state, tensor_action)[0]
        decoded = model.decode_tensor_state(tensor_next)
        ok += int(decoded == direct)
    return {"checked": total, "passed": ok}


def _aggregate(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    solved = [row[key]["solved"] for row in rows]
    pushes = [row[key]["solution_pushes"] for row in rows if row[key]["solution_pushes"] is not None]
    expanded = [row[key]["expanded"] for row in rows]
    elapsed = [row[key]["elapsed_sec"] for row in rows]
    return {
        "solved": int(sum(solved)),
        "count": len(rows),
        "solve_rate": float(sum(solved) / max(len(rows), 1)),
        "mean_solution_pushes": float(mean(pushes)) if pushes else None,
        "mean_expanded": float(mean(expanded)) if expanded else 0.0,
        "mean_elapsed_sec": float(mean(elapsed)) if elapsed else 0.0,
    }


def main() -> None:
    args = build_parser().parse_args()
    levels = load_boxoban_levels(args.levels, max_levels=args.max_levels)
    if not levels:
        raise SystemExit(f"No levels loaded from {args.levels!r}.")

    rows: list[dict[str, Any]] = []
    consistency_checked = 0
    consistency_passed = 0

    for idx, level in enumerate(levels):
        model = SokobanGoalDirectedSDModel(level)
        features = model.abstract_features(model.initial_state)
        consistency = _transition_consistency(model)
        consistency_checked += consistency["checked"]
        consistency_passed += consistency["passed"]

        unguided = best_first_search(
            model,
            max_expansions=args.max_expansions,
            heuristic_weight=0.0,
            prune_deadlocks=False,
        )
        sd_guided = best_first_search(
            model,
            max_expansions=args.max_expansions,
            heuristic_weight=args.heuristic_weight,
            prune_deadlocks=True,
        )
        rows.append(
            {
                "index": idx,
                "level_id": level.level_id,
                "initial_features": features.to_dict(),
                "transition_consistency": consistency,
                "unguided_uniform": unguided.to_dict(),
                "sd_guided": sd_guided.to_dict(),
            }
        )

    summary = {
        "config": {
            "levels": os.path.relpath(args.levels, ROOT) if os.path.isabs(args.levels) else args.levels,
            "max_levels": args.max_levels,
            "max_expansions": args.max_expansions,
            "heuristic_weight": args.heuristic_weight,
        },
        "transition_consistency": {
            "checked": consistency_checked,
            "passed": consistency_passed,
            "pass_rate": float(consistency_passed / max(consistency_checked, 1)),
        },
        "aggregate": {
            "unguided_uniform": _aggregate(rows, "unguided_uniform"),
            "sd_guided": _aggregate(rows, "sd_guided"),
        },
        "levels": rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2, ensure_ascii=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
