"""Shared utilities for offline Hopper retrospective analysis."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


DEFAULT_RESULTS_ROOT = Path("results/sd_retrospective_rule_induction")
DEFAULT_MANIFEST = DEFAULT_RESULTS_ROOT / "manifests" / "hopper_offline_v1_manifest.json"
DEFAULT_SELECTION = DEFAULT_RESULTS_ROOT / "manifests" / "hopper_offline_v1_selected_trajectories.json"
DEFAULT_ATTRIBUTIONS = DEFAULT_RESULTS_ROOT / "attributions" / "hopper_offline_v1_attributions.json"
DEFAULT_COUNTERFACTUALS = DEFAULT_RESULTS_ROOT / "attributions" / "hopper_offline_v1_counterfactuals.json"
DEFAULT_EXPLAINER = DEFAULT_RESULTS_ROOT / "sd_explainers" / "hopper_offline_v2_sd_explainer.pt"
DEFAULT_EXPLAINER_SUMMARY = DEFAULT_RESULTS_ROOT / "sd_explainers" / "hopper_offline_v2_sd_explainer_summary.json"
DEFAULT_COUNTERFACTUALS_V2 = DEFAULT_RESULTS_ROOT / "attributions" / "hopper_offline_v2_counterfactuals_explainer.json"
DEFAULT_RULES_V2 = DEFAULT_RESULTS_ROOT / "rules" / "hopper_offline_v2_state_action_rules.json"
DEFAULT_RULE_CONSISTENCY_V3 = DEFAULT_RESULTS_ROOT / "reports" / "hopper_offline_v3_rule_consistency.json"
DEFAULT_RULES_V3 = DEFAULT_RESULTS_ROOT / "rules" / "hopper_offline_v3_reliable_rules.json"
DEFAULT_SUBSTATE_RULES_V4 = DEFAULT_RESULTS_ROOT / "rules" / "hopper_offline_v4_substate_rules.json"
DEFAULT_REPLAY_SCORING_V4 = DEFAULT_RESULTS_ROOT / "reports" / "hopper_offline_v4_replay_rule_scoring.json"
DEFAULT_RULE_LIBRARY = DEFAULT_RESULTS_ROOT / "rules" / "library.json"
DEFAULT_SUBSTATE_VALIDATION_V5 = DEFAULT_RESULTS_ROOT / "reports" / "hopper_offline_v5_substate_rule_validation.json"
DEFAULT_RULE_LIBRARY_V5 = DEFAULT_RESULTS_ROOT / "rules" / "library_v5.json"
DEFAULT_RULES = DEFAULT_RESULTS_ROOT / "rules" / "hopper_offline_v1_rules.json"
DEFAULT_VALIDATION = DEFAULT_RESULTS_ROOT / "reports" / "hopper_offline_v1_rule_validation.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    weight = pos - lo
    return values[lo] * (1.0 - weight) + values[hi] * weight


def mean_or_zero(values: Iterable[float]) -> float:
    vals = list(values)
    return float(mean(vals)) if vals else 0.0


def parse_step_dir(step_dir: str) -> int:
    return int(step_dir.replace("step_", ""))


def repo_path(repo_root: Path, maybe_path: str | None) -> Path | None:
    if not maybe_path:
        return None
    path = Path(maybe_path)
    if path.is_absolute():
        return path
    return repo_root / path


def action_abs_mean(row: dict[str, str]) -> float:
    vals = [abs(f(row, f"action_{i}", 0.0)) for i in range(4)]
    return sum(vals) / len(vals)


def inv_contact(x: float) -> float:
    x = min(max(x, 0.0), 1.0 - 1e-6)
    return -math.log(max(1.0 - x, 1e-6))


def row_to_raw_obs(row: dict[str, str]) -> list[float]:
    return [
        f(row, "qpos_rootz"),
        f(row, "qpos_rooty"),
        f(row, "qpos_waist"),
        f(row, "qpos_hip"),
        f(row, "qpos_knee"),
        f(row, "qpos_ankle"),
        f(row, "qvel_rootx"),
        f(row, "qvel_rootz"),
        f(row, "qvel_rooty"),
        f(row, "qvel_waist"),
        f(row, "qvel_hip"),
        f(row, "qvel_knee"),
        f(row, "qvel_ankle"),
        inv_contact(f(row, "touch_toe")),
        inv_contact(f(row, "touch_heel")),
    ]


def row_to_action(row: dict[str, str]) -> list[float]:
    return [f(row, f"action_{i}") for i in range(4)]


def label_episode(reward: float, bad_threshold: float, good_threshold: float) -> str:
    if reward <= bad_threshold:
        return "bad"
    if reward >= good_threshold:
        return "good"
    return "middle"


def safe_rate(count: int, total: int) -> float:
    return float(count) / float(total) if total else 0.0


def resolve_workers(requested: int | None = 0, env_var: str = "SD_RETRO_WORKERS", default_cap: int = 16) -> int:
    """Resolve bounded CPU worker count for retrospective post-processing.

    A value of 0 or None means "auto". The environment variable can override
    auto mode so running experiments can be throttled without changing launch
    scripts. The default cap intentionally avoids consuming all logical cores.
    """

    if requested is not None and requested > 0:
        return int(requested)
    env_value = os.environ.get(env_var)
    if env_value:
        try:
            env_workers = int(env_value)
        except ValueError:
            env_workers = 0
        if env_workers > 0:
            return env_workers
    return max(1, min(default_cap, os.cpu_count() or 1))
