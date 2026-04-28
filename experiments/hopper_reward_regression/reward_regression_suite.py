from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "experiments" / "hopper_reward_regression" / "results"


@dataclass(frozen=True)
class StateCase:
    name: str
    relative_csv: str
    step: int
    expected_state: int
    note: str


@dataclass(frozen=True)
class TransitionCase:
    name: str
    relative_csv: str
    prev_step: int
    next_step: int
    expected_prev_state: int
    expected_next_state: int
    reward_assertion: str
    reward_threshold: float
    note: str


STATE_CASES = [
    StateCase(
        name="fake_rear_now_prone",
        relative_csv=(
            "logs/hopper-hop/3/"
            "sd_hopper_100000_v5densepose_recover010_seed3_20260426_run1/"
            "official_eval/step_0100000/episode_csv/episode_01.csv"
        ),
        step=98,
        expected_state=2,
        note="前伏抬身片段，旧口径会误记为 rear。",
    ),
    StateCase(
        name="true_rear_stays_rear",
        relative_csv=(
            "logs/hopper-hop/3/"
            "sd_hopper_100000_v5densepose_recover010_seed3_20260426_run1/"
            "official_eval/step_0100000/episode_csv/episode_02.csv"
        ),
        step=50,
        expected_state=1,
        note="真正整体后仰的失败片段，应保持 rear。",
    ),
    StateCase(
        name="upright_sample_kept_upright",
        relative_csv=(
            "logs/hopper-hop/2/"
            "sd_hopper_50000_v5densepose_alpha5to05_seed2_20260426_run1/"
            "official_eval/step_0050000/episode_csv/episode_09.csv"
        ),
        step=11,
        expected_state=3,
        note="短暂直立样例，应继续判为 upright。",
    ),
    StateCase(
        name="rear_landing_with_legs_airborne_is_rear",
        relative_csv=(
            "logs/hopper-hop/3/"
            "sd_hopper_10k_v5geomrear_smoke_20260426_run1/"
            "official_eval/step_0005000/episode_csv/episode_06.csv"
        ),
        step=11,
        expected_state=1,
        note="后仰躺地但腿脚悬空的着地片段，应判为 rear 而不是 prone。",
    ),
    StateCase(
        name="stable_rear_with_legs_airborne_stays_rear",
        relative_csv=(
            "logs/hopper-hop/3/"
            "sd_hopper_10k_v5geomrear_smoke_20260426_run1/"
            "official_eval/step_0005000/episode_csv/episode_06.csv"
        ),
        step=100,
        expected_state=1,
        note="后仰躺地稳定阶段，不应因脚悬空而在 rear/prone 间抖动。",
    ),
]


TRANSITION_CASES = [
    TransitionCase(
        name="prone_to_upright_positive_reward",
        relative_csv=(
            "logs/hopper-hop/2/"
            "sd_hopper_50000_v5densepose_alpha5to05_seed2_20260426_run1/"
            "official_eval/step_0025000/episode_csv/episode_02.csv"
        ),
        prev_step=10,
        next_step=11,
        expected_prev_state=2,
        expected_next_state=3,
        reward_assertion="gt",
        reward_threshold=0.05,
        note="命中 prone->upright 后，shaping 应为正。",
    ),
    TransitionCase(
        name="falling_to_prone_not_positive",
        relative_csv=(
            "logs/hopper-hop/3/"
            "sd_hopper_100000_v5densepose_recover010_seed3_20260426_run1/"
            "official_eval/step_0100000/episode_csv/episode_01.csv"
        ),
        prev_step=9,
        next_step=10,
        expected_prev_state=0,
        expected_next_state=2,
        reward_assertion="le",
        reward_threshold=0.0,
        note="falling->prone 不是目标迁移，不应拿正向里程碑奖励。",
    ),
]


def _model() -> DMControlHopperSDModel:
    model = DMControlHopperSDModel()
    model.eval()
    return model


def _csv_rows(relative_csv: str) -> list[dict[str, str]]:
    path = REPO_ROOT / relative_csv
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row_to_state(row: dict[str, str]) -> torch.Tensor:
    values = [
        float(row["torso_z"]),
        float(row["vx"]),
        float(row["vz"]),
        float(row["pitch"]),
        float(row["pitch_rate"]),
        float(row["activation"]),
        float(row["qpos_waist"]),
        float(row["qpos_hip"]),
        float(row["qpos_knee"]),
        float(row["qpos_ankle"]),
        float(row["qvel_waist"]),
        float(row["qvel_hip"]),
        float(row["qvel_knee"]),
        float(row["qvel_ankle"]),
        float(row["touch_toe"]),
        float(row["touch_heel"]),
        float(row["z_min"]),
    ]
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)


def _state_features(model: DMControlHopperSDModel, row: dict[str, str]) -> dict[str, Any]:
    state = _row_to_state(row)
    feat = model._state_reward_features(state)
    state_id = int(model._dominant_state_ids(feat)[0].item())
    return {
        "state_id": state_id,
        "pitch": float(feat["pitch"][0].item()),
        "nose_minus_foot_x": float(feat["nose_minus_foot_x"][0].item()),
        "nose_minus_anchor_x": float(feat["nose_minus_anchor_x"][0].item()),
        "posture_anchor_x": float(feat["posture_anchor_x"][0].item()),
        "body_contact_x": float(feat["body_contact_x"][0].item()),
        "nose_minus_support_x": float(feat["nose_minus_support_x"][0].item()),
        "hip_minus_support_x": float(feat["hip_minus_support_x"][0].item()),
        "full_foot_support": float(feat["full_foot_support"][0].item()),
        "body_collision": float(feat["body_collision"][0].item()),
        "upright_quality_gate": float(feat["upright_quality_gate"][0].item()),
        "prepared_quality_gate": float(feat["prepared_quality_gate"][0].item()),
    }


def evaluate_state_case(case: StateCase) -> dict[str, Any]:
    model = _model()
    rows = _csv_rows(case.relative_csv)
    feat = _state_features(model, rows[case.step])
    passed = feat["state_id"] == case.expected_state
    return {
        "kind": "state",
        "name": case.name,
        "passed": passed,
        "expected_state": case.expected_state,
        "observed_state": feat["state_id"],
        "step": case.step,
        "relative_csv": case.relative_csv,
        "note": case.note,
        "features": feat,
    }


def evaluate_transition_case(case: TransitionCase) -> dict[str, Any]:
    model = _model()
    rows = _csv_rows(case.relative_csv)
    prev_row = rows[case.prev_step]
    next_row = rows[case.next_step]
    prev_state = _state_features(model, prev_row)
    next_state = _state_features(model, next_row)
    prev_tensor = _row_to_state(prev_row)
    next_tensor = _row_to_state(next_row)
    reward = float(
        model.predict_shaping_reward(
            prev_tensor,
            torch.zeros((1, model.ACTION_DIM), dtype=torch.float32),
            next_tensor,
        )[0].item()
    )
    state_ok = (
        prev_state["state_id"] == case.expected_prev_state
        and next_state["state_id"] == case.expected_next_state
    )
    if case.reward_assertion == "gt":
        reward_ok = reward > case.reward_threshold
    elif case.reward_assertion == "ge":
        reward_ok = reward >= case.reward_threshold
    elif case.reward_assertion == "lt":
        reward_ok = reward < case.reward_threshold
    elif case.reward_assertion == "le":
        reward_ok = reward <= case.reward_threshold
    else:
        raise ValueError(f"unknown reward_assertion: {case.reward_assertion}")
    return {
        "kind": "transition",
        "name": case.name,
        "passed": state_ok and reward_ok,
        "expected_prev_state": case.expected_prev_state,
        "observed_prev_state": prev_state["state_id"],
        "expected_next_state": case.expected_next_state,
        "observed_next_state": next_state["state_id"],
        "reward_assertion": case.reward_assertion,
        "reward_threshold": case.reward_threshold,
        "observed_reward": reward,
        "prev_step": case.prev_step,
        "next_step": case.next_step,
        "relative_csv": case.relative_csv,
        "note": case.note,
        "prev_features": prev_state,
        "next_features": next_state,
    }


def run_suite() -> dict[str, Any]:
    state_results = [evaluate_state_case(case) for case in STATE_CASES]
    transition_results = [evaluate_transition_case(case) for case in TRANSITION_CASES]
    results = state_results + transition_results
    passed = sum(1 for item in results if item["passed"])
    return {
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
        },
        "state_cases": [asdict(case) for case in STATE_CASES],
        "transition_cases": [asdict(case) for case in TRANSITION_CASES],
        "results": results,
        "coverage_note": (
            "当前真实轨迹中尚未检出稳定的 upright->prepared、prepared->rising、"
            "falling->prepared 正样例，因此本轮先锁定真假 rear 与 prone->upright 的语义回归。"
        ),
    }


def write_report(report: dict[str, Any], out_path: Path | None = None) -> Path:
    if out_path is None:
        out_path = RESULTS_DIR / "latest_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return out_path
