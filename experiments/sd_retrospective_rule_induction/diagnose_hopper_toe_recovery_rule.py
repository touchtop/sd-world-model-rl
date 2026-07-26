#!/usr/bin/env python3
"""Diagnose whether SD supports a toe-only recovery-delay Hopper rule.

The script is intentionally offline and read-only with respect to training
artifacts. It tests a specific methodological question: when high-score Hopper
traces show toe-only force production from low or prone recovery states, does
the fitted SD explainer prefer actions whose predicted effect is continued
full-foot/support recovery over actions that keep producing early release?
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import f, read_csv, row_to_action, row_to_raw_obs, write_json  # noqa: E402
from counterfactual_actions import build_action_bank  # noqa: E402
from counterfactual_hopper_v8_transition_actions import load_model  # noqa: E402


DEFAULT_RUN_ROOT = Path(
    "results/sd_retrospective_rule_induction/closed_loop/"
    "hopper_v8formal_100k_confidence_candidate_seed123_20260510_r3/cycle_020"
)
DEFAULT_RUN_NAME = "tdmpc2_v8formal_100k_confidence_candidate_seed123_20260510_r3_cycle020_target100019"
DEFAULT_SD_CHECKPOINT = DEFAULT_RUN_ROOT / "sd_explainers" / "sd_explainer_cycle_020.pt"
DEFAULT_LOG_ROOT = Path("logs/hopper-hop")
DEFAULT_OUTPUT = (
    Path("results/sd_retrospective_rule_induction/reports")
    / "hopper_100k_toe_recovery_sd_diagnostic.json"
)


def path_from_summary(row: dict[str, str]) -> Path:
    raw = row.get("trace_csv", "")
    path = Path(raw)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def episode_return(path: Path) -> float:
    return sum(f(row, "reward") for row in read_csv(path))


def discover_top_traces(
    log_root: Path,
    run_name: str,
    seeds: list[int],
    top_k: int,
    include_official: bool,
    include_replay: bool,
) -> list[dict[str, Any]]:
    trace_specs: list[dict[str, Any]] = []
    sources: list[tuple[str, str]] = []
    if include_official:
        sources.append(("official_eval", "step_0100019"))
    if include_replay:
        sources.append(("retrospective_train_traces", "step_0100020"))

    for seed in seeds:
        for kind, step_dir in sources:
            root = log_root / str(seed) / run_name / kind / step_dir
            summary = root / "eval_episode_summary.csv"
            if summary.exists():
                with summary.open("r", newline="") as fp:
                    for row in csv.DictReader(fp):
                        path = path_from_summary(row)
                        if not path.exists():
                            continue
                        trace_specs.append(
                            {
                                "seed": seed,
                                "source": kind,
                                "episode": row.get("episode_idx"),
                                "return": float(row.get("episode_reward", 0.0) or 0.0),
                                "trace_csv": str(path),
                            }
                        )
                continue

            episode_dir = root / "episode_csv"
            if not episode_dir.exists():
                continue
            for path in sorted(episode_dir.glob("episode_*.csv")):
                trace_specs.append(
                    {
                        "seed": seed,
                        "source": kind,
                        "episode": path.stem.replace("episode_", ""),
                        "return": episode_return(path),
                        "trace_csv": str(path.resolve()),
                    }
                )

    trace_specs.sort(key=lambda item: item["return"], reverse=True)
    return trace_specs[:top_k]


def is_toe_recovery_state(row: dict[str, str]) -> bool:
    toe_ready = f(row, "touch_toe") >= 0.80
    heel_absent = f(row, "touch_heel") < 0.50
    no_full_support = f(row, "full_foot_support") < 0.35
    not_prepared = f(row, "grounded_prepared") < 0.50 and f(row, "grounded_upright") < 0.50
    low_or_prone = (
        f(row, "grounded_prone") >= 0.50
        or f(row, "low_body") >= 0.50
        or f(row, "torso_z", f(row, "rootz")) <= 0.72
        or f(row, "rotation_penalty") >= 0.50
    )
    return toe_ready and heel_absent and no_full_support and not_prepared and low_or_prone


def is_release_next(row: dict[str, str]) -> bool:
    return (
        f(row, "airborne_true") >= 0.50
        or (
            f(row, "touch_toe") < 0.50
            and f(row, "touch_heel") < 0.50
            and f(row, "foot_grounded") < 0.50
        )
    )


def observed_window(rows: list[dict[str, str]], idx: int, horizon: int) -> dict[str, Any]:
    post = rows[idx + 1 : min(len(rows), idx + 1 + horizon)]
    nxt = rows[idx + 1] if idx + 1 < len(rows) else None
    return {
        "next_release": bool(nxt is not None and is_release_next(nxt)),
        "post_prone": any(f(row, "grounded_prone") >= 0.50 for row in post),
        "post_falling_air": any(f(row, "falling_air") >= 0.50 for row in post),
        "post_full_foot_or_prepared": any(
            f(row, "full_foot_support") >= 0.50
            or f(row, "grounded_prepared") >= 0.50
            or f(row, "grounded_upright") >= 0.50
            for row in post
        ),
        "max_full_foot": max([f(row, "full_foot_support") for row in post] or [0.0]),
        "max_prepared": max([f(row, "grounded_prepared") for row in post] or [0.0]),
        "max_prone": max([f(row, "grounded_prone") for row in post] or [0.0]),
        "min_torso_z": min([f(row, "torso_z", f(row, "rootz")) for row in post] or [0.0]),
        "max_rotation_penalty": max([f(row, "rotation_penalty") for row in post] or [0.0]),
    }


def action_family(name: str) -> str:
    parts = name.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else name


def family_from_action(action: list[float], threshold: float) -> str:
    names = ("waist", "hip", "knee", "ankle")
    idx, value = max(enumerate(action), key=lambda item: abs(item[1]))
    if abs(value) < threshold:
        return "all_weak"
    return f"{names[idx]}_{'pos' if value > 0 else 'neg'}"


def recovery_score(feat: dict[str, torch.Tensor]) -> torch.Tensor:
    """Score delayed recovery rather than immediate ballistic release."""

    support = torch.maximum(feat["full_foot_support"], feat["support_contact"])
    prepared = torch.maximum(feat["grounded_prepared"], feat["grounded_upright"])
    torso = torch.clamp((feat["torso_z"] - 0.35) / 0.55, min=0.0, max=1.0)
    controlled_spin = 1.0 - torch.clamp(torch.abs(feat["pitch_rate"]) / 5.0, min=0.0, max=1.0)
    forward = torch.clamp(feat["vx"] / 2.0, min=0.0, max=1.0)
    bad_pose = torch.maximum(feat["grounded_prone"], feat["grounded_rear"])
    falling = feat["falling_air"]
    rotation = torch.clamp(feat["rotation_penalty"], min=0.0, max=1.0)
    # The weights express the research hypothesis: support recovery should
    # dominate short-horizon forward speed when the agent is still toe-only.
    return (
        1.40 * support
        + 1.10 * prepared
        + 0.65 * torso
        + 0.45 * controlled_spin
        + 0.20 * forward
        - 1.15 * bad_pose
        - 0.80 * falling
        - 0.65 * rotation
    )


def summarize_features(feat: dict[str, torch.Tensor], idx: int) -> dict[str, float]:
    keys = [
        "torso_z",
        "vx",
        "vz",
        "pitch",
        "pitch_rate",
        "support_contact",
        "full_foot_support",
        "grounded_prone",
        "grounded_rear",
        "grounded_upright",
        "grounded_prepared",
        "falling_air",
        "rotation_penalty",
    ]
    return {key: float(feat[key][idx].item()) for key in keys}


def sd_rollout(
    model: Any,
    row: dict[str, str],
    actions: torch.Tensor,
    horizon: int,
) -> tuple[list[float], list[dict[str, float]]]:
    state = model.encode_observation(torch.tensor(row_to_raw_obs(row), dtype=torch.float32).unsqueeze(0))
    cur = state.repeat(actions.shape[0], 1)
    with torch.no_grad():
        for _ in range(horizon):
            cur = model(cur, actions)
        feat = model._state_reward_features(cur)
        scores = recovery_score(feat)
    return (
        [float(x) for x in scores.detach().cpu().tolist()],
        [summarize_features(feat, idx) for idx in range(actions.shape[0])],
    )


def evaluate_state(
    model: Any,
    row: dict[str, str],
    observed: dict[str, Any],
    action_bank: list[tuple[str, torch.Tensor]],
    horizon: int,
    action_threshold: float,
) -> dict[str, Any]:
    actual = torch.tensor(row_to_action(row), dtype=torch.float32).unsqueeze(0).clamp(-1.0, 1.0)
    names = ["actual"] + [name for name, _ in action_bank]
    actions = torch.cat([actual, *[action for _, action in action_bank]], dim=0)
    scores, features = sd_rollout(model, row, actions, horizon)

    alternatives: list[dict[str, Any]] = []
    for name, score, summary in zip(names[1:], scores[1:], features[1:]):
        alternatives.append(
            {
                "name": name,
                "family": action_family(name),
                "score": score,
                "summary": summary,
            }
        )
    alternatives.sort(key=lambda item: item["score"], reverse=True)
    best = alternatives[0]
    actual_summary = features[0]
    best_summary = best["summary"]
    support_gain = best_summary["full_foot_support"] - actual_summary["full_foot_support"]
    prepared_gain = max(best_summary["grounded_prepared"], best_summary["grounded_upright"]) - max(
        actual_summary["grounded_prepared"], actual_summary["grounded_upright"]
    )
    prone_drop = actual_summary["grounded_prone"] - best_summary["grounded_prone"]
    falling_drop = actual_summary["falling_air"] - best_summary["falling_air"]
    rotation_drop = actual_summary["rotation_penalty"] - best_summary["rotation_penalty"]

    return {
        "step": int(f(row, "step")),
        "row_state": {
            "touch_toe": f(row, "touch_toe"),
            "touch_heel": f(row, "touch_heel"),
            "full_foot_support": f(row, "full_foot_support"),
            "grounded_prone": f(row, "grounded_prone"),
            "grounded_prepared": f(row, "grounded_prepared"),
            "grounded_upright": f(row, "grounded_upright"),
            "torso_z": f(row, "torso_z", f(row, "rootz")),
            "rotation_penalty": f(row, "rotation_penalty"),
            "vx": f(row, "vx"),
            "vz": f(row, "vz"),
            "pitch": f(row, "pitch"),
            "pitch_rate": f(row, "pitch_rate"),
        },
        "observed_window": observed,
        "actual_action": row_to_action(row),
        "actual_action_family": family_from_action(row_to_action(row), action_threshold),
        "actual_score": scores[0],
        "actual_summary": actual_summary,
        "best_action": best["name"],
        "best_action_family": best["family"],
        "best_score": best["score"],
        "advantage_over_actual": best["score"] - scores[0],
        "actual_rank_among_probe_actions": 1 + sum(1 for item in alternatives if item["score"] > scores[0]),
        "support_gain_vs_actual": support_gain,
        "prepared_gain_vs_actual": prepared_gain,
        "prone_drop_vs_actual": prone_drop,
        "falling_drop_vs_actual": falling_drop,
        "rotation_drop_vs_actual": rotation_drop,
        "sd_supports_recovery_delay": bool(
            best["score"] - scores[0] >= 0.05
            and (support_gain >= 0.05 or prepared_gain >= 0.05 or prone_drop >= 0.05 or rotation_drop >= 0.05)
        ),
        "top_actions": alternatives[:6],
    }


def select_states(rows: list[dict[str, str]], max_states: int, horizon: int) -> list[tuple[int, dict[str, Any]]]:
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    for idx, row in enumerate(rows[:-1]):
        if not is_toe_recovery_state(row):
            continue
        observed = observed_window(rows, idx, horizon)
        priority = (
            1.00 * float(observed["next_release"])
            + 0.75 * float(observed["post_prone"])
            + 0.50 * float(observed["post_falling_air"])
            + 0.30 * f(row, "grounded_prone")
            + 0.20 * f(row, "rotation_penalty")
        )
        candidates.append((priority, idx, observed))
    candidates.sort(key=lambda item: item[0], reverse=True)
    chosen = sorted(candidates[:max_states], key=lambda item: item[1])
    return [(idx, observed) for _, idx, observed in chosen]


def aggregate(records: list[dict[str, Any]], min_support: int, min_support_rate: float) -> dict[str, Any]:
    family_counts = Counter(rec["best_action_family"] for rec in records if rec["sd_supports_recovery_delay"])
    actual_family_counts = Counter(rec["actual_action_family"] for rec in records)
    supported = [rec for rec in records if rec["sd_supports_recovery_delay"]]
    release_records = [rec for rec in records if rec["observed_window"]["next_release"]]
    prone_records = [rec for rec in records if rec["observed_window"]["post_prone"]]
    recovery_records = [rec for rec in records if rec["observed_window"]["post_full_foot_or_prepared"]]
    support_rate = len(supported) / len(records) if records else 0.0

    top_families = [
        {
            "action_family": family,
            "support": count,
            "support_rate": count / len(records) if records else 0.0,
            "mean_advantage_over_actual": mean(
                rec["advantage_over_actual"]
                for rec in records
                if rec["sd_supports_recovery_delay"] and rec["best_action_family"] == family
            ),
        }
        for family, count in family_counts.most_common()
    ]
    candidate_rule = None
    if len(supported) >= min_support and support_rate >= min_support_rate:
        candidate_rule = {
            "id": "toe_recovery_low_support->delay_release_and_restore_full_foot_support",
            "rule_family": "hopper-stage-transition-template",
            "condition": {
                "touch_toe": ">=0.80",
                "touch_heel": "<0.50",
                "full_foot_support": "<0.35",
                "grounded_prepared/upright": "<0.50",
                "recovery_risk": "grounded_prone or low torso or rotation risk",
            },
            "preferred_effect": [
                "increase support_contact/full_foot_support",
                "increase grounded_prepared or grounded_upright",
                "reduce grounded_prone/falling_air/rotation_penalty",
            ],
            "avoid_effect": [
                "toe-only release before support recovery",
                "falling/prone continuation after toe release",
            ],
            "status": "offline_sd_diagnostic_candidate",
            "support": len(supported),
            "support_rate": support_rate,
            "top_preferred_action_families": top_families[:5],
            "mean_advantage_over_actual": mean(rec["advantage_over_actual"] for rec in supported),
            "mean_support_gain_vs_actual": mean(rec["support_gain_vs_actual"] for rec in supported),
            "mean_prepared_gain_vs_actual": mean(rec["prepared_gain_vs_actual"] for rec in supported),
            "mean_prone_drop_vs_actual": mean(rec["prone_drop_vs_actual"] for rec in supported),
            "mean_rotation_drop_vs_actual": mean(rec["rotation_drop_vs_actual"] for rec in supported),
        }

    return {
        "num_records": len(records),
        "num_sd_supported": len(supported),
        "sd_supported_rate": support_rate,
        "observed_next_release_rate": len(release_records) / len(records) if records else 0.0,
        "observed_post_prone_rate": len(prone_records) / len(records) if records else 0.0,
        "observed_post_recovery_rate": len(recovery_records) / len(records) if records else 0.0,
        "actual_action_family_counts": dict(actual_family_counts.most_common()),
        "sd_supported_best_action_family_counts": dict(family_counts.most_common()),
        "top_preferred_action_families": top_families,
        "candidate_rule": candidate_rule,
    }


def write_markdown(path: Path, payload: dict[str, Any], max_examples: int) -> None:
    agg = payload["aggregate"]
    rule = agg.get("candidate_rule")
    lines = [
        "# Hopper 100k Toe-Recovery SD Diagnostic",
        "",
        f"SD checkpoint: `{payload['model_checkpoint']}`",
        f"Trace source count: `{len(payload['traces'])}`",
        f"Diagnostic states: `{agg['num_records']}`",
        f"Observed next-release rate: `{agg['observed_next_release_rate']:.3f}`",
        f"Observed post-prone rate: `{agg['observed_post_prone_rate']:.3f}`",
        f"Observed post-recovery rate: `{agg['observed_post_recovery_rate']:.3f}`",
        f"SD-supported recovery-delay rate: `{agg['sd_supported_rate']:.3f}`",
        "",
        "## Candidate Rule",
        "",
    ]
    if rule is None:
        lines.append("No candidate rule passed the configured support thresholds.")
    else:
        lines.extend(
            [
                f"Rule id: `{rule['id']}`",
                "",
                f"- Status: `{rule['status']}`",
                f"- Support / rate: `{rule['support']}` / `{rule['support_rate']:.3f}`",
                f"- Mean advantage over actual: `{rule['mean_advantage_over_actual']:.4f}`",
                f"- Mean support gain: `{rule['mean_support_gain_vs_actual']:.4f}`",
                f"- Mean prepared gain: `{rule['mean_prepared_gain_vs_actual']:.4f}`",
                f"- Mean prone drop: `{rule['mean_prone_drop_vs_actual']:.4f}`",
                f"- Mean rotation drop: `{rule['mean_rotation_drop_vs_actual']:.4f}`",
                f"- Top preferred action families: `{rule['top_preferred_action_families']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Trace Inputs",
            "",
        ]
    )
    for item in payload["traces"]:
        lines.append(
            f"- seed `{item['seed']}`, `{item['source']}`, episode `{item['episode']}`, "
            f"return `{item['return']:.3f}`: `{item['trace_csv']}`"
        )
    lines.extend(["", "## Examples", ""])
    for rec in payload["records"][:max_examples]:
        lines.extend(
            [
                f"### `{rec['trace_csv']}` step `{rec['step']}`",
                "",
                f"- Observed: `{rec['observed_window']}`",
                f"- Actual family / score: `{rec['actual_action_family']}` / `{rec['actual_score']:.4f}`",
                f"- Best family / score: `{rec['best_action_family']}` / `{rec['best_score']:.4f}`",
                f"- Advantage: `{rec['advantage_over_actual']:.4f}`",
                f"- SD supports rule: `{rec['sd_supports_recovery_delay']}`",
                "",
            ]
        )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-checkpoint", type=Path, default=DEFAULT_SD_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--trace", type=Path, action="append", default=[])
    parser.add_argument("--top-k-traces", type=int, default=8)
    parser.add_argument("--official-only", action="store_true")
    parser.add_argument("--replay-only", action="store_true")
    parser.add_argument("--max-states-per-trace", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[0.35, 0.70, 1.00])
    parser.add_argument("--action-threshold", type=float, default=0.30)
    parser.add_argument("--min-support", type=int, default=8)
    parser.add_argument("--min-support-rate", type=float, default=0.35)
    parser.add_argument("--max-markdown-examples", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, loaded = load_model(args.model_checkpoint)
    action_bank = build_action_bank(args.amplitudes)

    traces: list[dict[str, Any]]
    if args.trace:
        traces = [
            {
                "seed": None,
                "source": "manual",
                "episode": path.stem,
                "return": episode_return(path),
                "trace_csv": str(path.resolve()),
            }
            for path in args.trace
        ]
    else:
        include_official = not args.replay_only
        include_replay = not args.official_only
        traces = discover_top_traces(
            log_root=args.log_root,
            run_name=args.run_name,
            seeds=args.seeds,
            top_k=args.top_k_traces,
            include_official=include_official,
            include_replay=include_replay,
        )

    records: list[dict[str, Any]] = []
    by_trace: dict[str, int] = defaultdict(int)
    for trace in traces:
        rows = read_csv(Path(trace["trace_csv"]))
        for idx, observed in select_states(rows, args.max_states_per_trace, args.horizon):
            rec = evaluate_state(
                model=model,
                row=rows[idx],
                observed=observed,
                action_bank=action_bank,
                horizon=args.horizon,
                action_threshold=args.action_threshold,
            )
            rec.update(
                {
                    "trace_csv": trace["trace_csv"],
                    "seed": trace["seed"],
                    "source": trace["source"],
                    "episode": trace["episode"],
                    "episode_return": trace["return"],
                    "row_index": idx,
                }
            )
            records.append(rec)
            by_trace[trace["trace_csv"]] += 1

    agg = aggregate(records, args.min_support, args.min_support_rate)
    payload = {
        "diagnostic_family": "hopper-toe-recovery-delay-sd-diagnostic",
        "model_checkpoint": loaded,
        "run_name": args.run_name,
        "horizon": args.horizon,
        "amplitudes": args.amplitudes,
        "state_predicate": "toe_only_low_or_prone_recovery_without_full_foot_support",
        "traces": traces,
        "states_per_trace": dict(by_trace),
        "aggregate": agg,
        "records": records,
    }
    write_json(args.output, payload)
    write_markdown(args.output, payload, args.max_markdown_examples)
    print(f"wrote diagnostic: {args.output}")
    print(f"states: {agg['num_records']} sd_supported: {agg['num_sd_supported']} rate={agg['sd_supported_rate']:.3f}")
    if agg["candidate_rule"] is not None:
        print(f"candidate_rule: {agg['candidate_rule']['id']}")
    else:
        print("candidate_rule: none")


if __name__ == "__main__":
    main()
