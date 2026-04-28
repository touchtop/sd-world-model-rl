#!/usr/bin/env python3
"""Diagnose falling->landing contact onset timing on Hopper official-eval traces."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TD_ROOT = REPO_ROOT / "external" / "tdmpc2" / "tdmpc2"
for p in (REPO_ROOT, TD_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.sd_world_model import SDWorldModel
from scripts.analyze_hopper_action_ranking_diagnostic import MiniCfg
from scripts.analyze_hopper_sd_action_sensitivity import _row_to_trace_state, iter_trace_paths
from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel


class DynamicsOnlyAdapter:
    def __init__(self, dynamics: DMControlHopperSDModel):
        self._dynamics = dynamics

    def encode(self, obs: torch.Tensor, _task) -> torch.Tensor:
        return self._dynamics.encode_observation(obs)

    def next(self, z: torch.Tensor, a: torch.Tensor, _task) -> torch.Tensor:
        return self._dynamics(z, a)


@dataclass
class OnsetSample:
    trace_path: str
    episode: int
    row_idx: int
    step: int
    raw_obs: np.ndarray
    future_actions: np.ndarray
    future_rows: List[dict[str, str]]


def _infer_agent_cfg_from_state_dict(state_dict: dict[str, torch.Tensor], args) -> MiniCfg:
    inferred_mlp_dim = int(state_dict["_pi.0.weight"].shape[0])
    inferred_latent_dim = int(state_dict["_pi.0.weight"].shape[1])
    inferred_num_bins = int(state_dict["_reward.2.weight"].shape[0])
    inferred_num_q = len(
        {
            int(k.split(".")[2])
            for k in state_dict
            if k.startswith("_Qs.modules_list.") and k.split(".")[3] == "0"
        }
    )
    cfg = MiniCfg(
        mlp_dim=inferred_mlp_dim,
        num_bins=inferred_num_bins,
        num_q=inferred_num_q,
        dropout=float(args.dropout),
        log_std_min=float(args.log_std_min),
        log_std_max=float(args.log_std_max),
        sd_struct_dim=int(args.sd_struct_dim),
        sd_latent_dim=inferred_latent_dim,
        latent_dim=inferred_latent_dim,
        sd_hidden_dim=int(args.sd_hidden_dim),
        sd_hop_speed=float(args.sd_hop_speed),
        sd_transition_residual_scale=float(args.sd_transition_residual_scale),
        discount_max=float(args.discount),
    )
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    return cfg


def build_world_model(args):
    payload = torch.load(args.agent_checkpoint, map_location="cpu", weights_only=False)
    if "model" in payload:
        state_dict = payload["model"]
        cfg = _infer_agent_cfg_from_state_dict(state_dict, args)
        model = SDWorldModel(cfg)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            missing_touch_only = True
            missing = []
            unexpected = []
            text = str(exc)
            for line in text.splitlines():
                if "Missing key(s) in state_dict:" in line:
                    missing = [
                        part.strip().strip('"')
                        for part in line.split(":", 1)[1].split(",")
                        if part.strip()
                    ]
                if "Unexpected key(s) in state_dict:" in line:
                    unexpected = [
                        part.strip().strip('"')
                        for part in line.split(":", 1)[1].split(",")
                        if part.strip()
                    ]
            allowed_missing_prefixes = {
                "_dynamics.touch_transition_residual_head.",
                "_dynamics.touch_transition_residual_base_scale",
            }
            for key in missing:
                if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes):
                    missing_touch_only = False
                    break
            if unexpected:
                missing_touch_only = False
            if not missing_touch_only:
                raise
            model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    src_state = payload.get("model_state", payload)
    src_args = payload.get("args", {})
    dynamics = DMControlHopperSDModel(
        action_prior_enabled=False,
        shaping_scale=0.0,
        dense_pose_delta_weight=0.0,
        dense_pose_abs_weight=0.0,
        dense_pose_milestone_scale=0.0,
        recover_penalty_scale=0.0,
        stage_mode="full",
        use_transition_residual=bool(src_args.get("use_transition_residual", True)),
        transition_residual_scale=float(src_args.get("transition_residual_scale", args.sd_transition_residual_scale)),
    )
    target_state = dynamics.state_dict()
    compatible = {
        k: v for k, v in src_state.items()
        if k in target_state and tuple(v.shape) == tuple(target_state[k].shape)
    }
    dynamics.load_state_dict(compatible, strict=False)
    dynamics.eval()
    return DynamicsOnlyAdapter(dynamics)


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    return float(row.get(key, default) or default)


def _state_name(row: dict[str, str]) -> str:
    sid = int(float(row["dominant_state_id"]))
    if sid < 0 or sid >= len(DMControlHopperSDModel.STATE_NAMES):
        return f"state_{sid}"
    return DMControlHopperSDModel.STATE_NAMES[sid]


def _event_from_row(row: dict[str, str], event: str, threshold: float, ground_eps: float) -> bool:
    if event == "toe_touch":
        return _float(row, "touch_toe") >= threshold
    if event == "heel_touch":
        return _float(row, "touch_heel") >= threshold
    if event == "foot_grounded":
        return _float(row, "foot_grounded") >= threshold
    if event == "grounded_any":
        return _float(row, "any_contact_z") <= ground_eps
    if event == "airborne_false":
        return _float(row, "airborne_true") < 0.5
    raise KeyError(event)


def _event_from_feat(feat: Dict[str, torch.Tensor], event: str, threshold: float, ground_eps: float) -> bool:
    if event == "toe_touch":
        return float(feat["touch_toe"][0].item()) >= threshold
    if event == "heel_touch":
        return float(feat["touch_heel"][0].item()) >= threshold
    if event == "foot_grounded":
        return float(feat["foot_grounded"][0].item()) >= threshold
    if event == "grounded_any":
        return float(feat["any_contact_z"][0].item()) <= ground_eps
    if event == "airborne_false":
        return float(feat["airborne_true"][0].item()) < 0.5
    raise KeyError(event)


def _first_true_step(flags: Sequence[bool]) -> int | None:
    for i, flag in enumerate(flags, start=1):
        if flag:
            return i
    return None


def _sample_true_landing_within(
    rows: Sequence[dict[str, str]],
    row_idx: int,
    horizon: int,
    threshold: float,
    ground_eps: float,
) -> bool:
    future = rows[row_idx + 1 : row_idx + 1 + horizon]
    if not future:
        return False
    return any(_event_from_row(row, "grounded_any", threshold, ground_eps) for row in future)


def load_onset_samples(
    trace_paths: Iterable[str],
    samples_per_trace: int,
    horizon: int,
    min_step: int,
    max_step: int | None,
    state_name_filter: Sequence[str] | None,
    require_trace_landing_within: int | None,
    touch_threshold: float,
    ground_eps: float,
) -> List[OnsetSample]:
    out: List[OnsetSample] = []
    allowed_names = set(state_name_filter or [])
    for trace_path in trace_paths:
        with open(trace_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        candidates: List[int] = []
        for row_idx, row in enumerate(rows):
            step = int(float(row["step"]))
            if step < min_step:
                continue
            if max_step is not None and step > max_step:
                continue
            if int(float(row.get("done", 0) or 0)) != 0:
                continue
            if allowed_names and _state_name(row) not in allowed_names:
                continue
            if require_trace_landing_within is not None:
                if _float(row, "airborne_true") < 0.5:
                    continue
                if not _sample_true_landing_within(rows, row_idx, require_trace_landing_within, touch_threshold, ground_eps):
                    continue
            future_rows = rows[row_idx + 1 : row_idx + 1 + horizon]
            if len(future_rows) < horizon:
                continue
            if row_idx + horizon > len(rows) - 1:
                continue
            candidates.append(row_idx)
        if not candidates:
            continue
        keep = min(samples_per_trace, len(candidates))
        idxs = np.linspace(0, len(candidates) - 1, num=keep, dtype=int)
        for idx in idxs.tolist():
            row_idx = candidates[idx]
            row = rows[row_idx]
            sample = _row_to_trace_state(trace_path, row)
            future_actions = np.array(
                [
                    [
                        _float(rows[row_idx + h], "action_0"),
                        _float(rows[row_idx + h], "action_1"),
                        _float(rows[row_idx + h], "action_2"),
                        _float(rows[row_idx + h], "action_3"),
                    ]
                    for h in range(horizon)
                ],
                dtype=np.float32,
            )
            out.append(
                OnsetSample(
                    trace_path=trace_path,
                    episode=sample.episode,
                    row_idx=row_idx,
                    step=sample.step,
                    raw_obs=sample.raw_obs,
                    future_actions=future_actions,
                    future_rows=rows[row_idx + 1 : row_idx + 1 + horizon],
                )
            )
    if not out:
        raise RuntimeError("No onset samples matched the requested filters.")
    return out


def rollout_predicted_features(
    model: SDWorldModel,
    sample: OnsetSample,
) -> List[Dict[str, torch.Tensor]]:
    z = model.encode(torch.from_numpy(sample.raw_obs).float().unsqueeze(0), None)
    out: List[Dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for action in sample.future_actions:
            a = torch.from_numpy(action).float().unsqueeze(0)
            z = model.next(z, a, None)
            z_struct = z[..., : DMControlHopperSDModel.STATE_DIM]
            feat = model._dynamics._state_reward_features(z_struct)
            out.append({k: v.detach().cpu() for k, v in feat.items()})
    return out


def _error_stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.shape[0]),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def summarise_event(records: Sequence[dict], event: str, horizon: int) -> Dict[str, object]:
    per_h_acc = []
    for h in range(1, horizon + 1):
        hits = [1.0 if r[f"{event}_true_flags"][h - 1] == r[f"{event}_pred_flags"][h - 1] else 0.0 for r in records]
        per_h_acc.append(float(np.mean(hits)) if hits else 0.0)
    onset_errors = [
        float(r[f"{event}_pred_onset"] - r[f"{event}_true_onset"])
        for r in records
        if r[f"{event}_pred_onset"] is not None and r[f"{event}_true_onset"] is not None
    ]
    onset_pairs = [
        (r[f"{event}_true_onset"], r[f"{event}_pred_onset"])
        for r in records
    ]
    both_missing = sum(1 for t, p in onset_pairs if t is None and p is None)
    pred_missing = sum(1 for t, p in onset_pairs if t is not None and p is None)
    true_missing = sum(1 for t, p in onset_pairs if t is None and p is not None)
    exact = sum(1 for t, p in onset_pairs if t is not None and p is not None and t == p)
    early = sum(1 for t, p in onset_pairs if t is not None and p is not None and p < t)
    late = sum(1 for t, p in onset_pairs if t is not None and p is not None and p > t)
    return {
        "num_records": int(len(records)),
        "onset_error": _error_stats(onset_errors),
        "classification_accuracy_by_horizon": per_h_acc,
        "exact_match_rate": float(exact / len(records)) if records else 0.0,
        "pred_early_rate": float(early / len(records)) if records else 0.0,
        "pred_late_rate": float(late / len(records)) if records else 0.0,
        "both_missing_rate": float(both_missing / len(records)) if records else 0.0,
        "pred_missing_rate": float(pred_missing / len(records)) if records else 0.0,
        "spurious_pred_rate": float(true_missing / len(records)) if records else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Hopper falling->landing contact onset timing.")
    parser.add_argument("--agent-checkpoint", required=True, help="Path to an RL agent checkpoint (.pt).")
    parser.add_argument("--trace-csv", action="append", default=[], help="Path to an episode_XX.csv trace file.")
    parser.add_argument("--trace-dir", default=None, help="Directory containing episode_XX.csv files.")
    parser.add_argument("--out-dir", required=True, help="Directory for summary/output files.")
    parser.add_argument("--samples-per-trace", type=int, default=2)
    parser.add_argument("--min-step", type=int, default=0)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--require-trace-landing-within", type=int, default=6)
    parser.add_argument("--state-name-filter", nargs="*", default=["falling_air"])
    parser.add_argument("--touch-threshold", type=float, default=0.5)
    parser.add_argument("--ground-eps", type=float, default=DMControlHopperSDModel.STATE_GROUND_EPS)
    parser.add_argument("--mlp-dim", type=int, default=512)
    parser.add_argument("--num-bins", type=int, default=101)
    parser.add_argument("--num-q", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--log-std-min", type=float, default=-10.0)
    parser.add_argument("--log-std-max", type=float, default=2.0)
    parser.add_argument("--sd-struct-dim", type=int, default=DMControlHopperSDModel.STATE_DIM)
    parser.add_argument("--sd-latent-dim", type=int, default=DMControlHopperSDModel.STATE_DIM)
    parser.add_argument("--sd-hidden-dim", type=int, default=512)
    parser.add_argument("--sd-hop-speed", type=float, default=2.0)
    parser.add_argument("--sd-transition-residual-scale", type=float, default=4.0)
    parser.add_argument("--discount", type=float, default=0.995)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_paths = iter_trace_paths(args.trace_csv, args.trace_dir)
    model = build_world_model(args)
    samples = load_onset_samples(
        trace_paths=trace_paths,
        samples_per_trace=int(args.samples_per_trace),
        horizon=int(args.horizon),
        min_step=int(args.min_step),
        max_step=args.max_step,
        state_name_filter=args.state_name_filter,
        require_trace_landing_within=args.require_trace_landing_within,
        touch_threshold=float(args.touch_threshold),
        ground_eps=float(args.ground_eps),
    )

    event_names = ("toe_touch", "heel_touch", "grounded_any", "foot_grounded", "airborne_false")
    records: List[dict] = []
    err_vx: List[float] = []
    err_vz: List[float] = []
    err_omega: List[float] = []
    err_any_contact_z: List[float] = []
    err_touch_toe: List[float] = []
    err_touch_heel: List[float] = []

    for sample in samples:
        pred_feats = rollout_predicted_features(model, sample)
        record = {
            "trace_path": sample.trace_path,
            "episode": sample.episode,
            "step": sample.step,
        }
        for event in event_names:
            true_flags = [
                _event_from_row(row, event, float(args.touch_threshold), float(args.ground_eps))
                for row in sample.future_rows
            ]
            pred_flags = [
                _event_from_feat(feat, event, float(args.touch_threshold), float(args.ground_eps))
                for feat in pred_feats
            ]
            record[f"{event}_true_flags"] = true_flags
            record[f"{event}_pred_flags"] = pred_flags
            record[f"{event}_true_onset"] = _first_true_step(true_flags)
            record[f"{event}_pred_onset"] = _first_true_step(pred_flags)
        for h, (row, feat) in enumerate(zip(sample.future_rows, pred_feats), start=1):
            true_vx = _float(row, "vx")
            true_vz = _float(row, "vz")
            true_omega = _float(row, "pitch_rate")
            true_any_contact_z = _float(row, "any_contact_z")
            true_touch_toe = _float(row, "touch_toe")
            true_touch_heel = _float(row, "touch_heel")
            pred_vx = float(feat["vx"][0].item())
            pred_vz = float(feat["vz"][0].item())
            pred_omega = float(feat["pitch_rate"][0].item())
            pred_any_contact_z = float(feat["any_contact_z"][0].item())
            pred_touch_toe = float(feat["touch_toe"][0].item())
            pred_touch_heel = float(feat["touch_heel"][0].item())
            if record["grounded_any_true_onset"] is not None and abs(h - record["grounded_any_true_onset"]) <= 1:
                err_vx.append(abs(pred_vx - true_vx))
                err_vz.append(abs(pred_vz - true_vz))
                err_omega.append(abs(pred_omega - true_omega))
                err_any_contact_z.append(abs(pred_any_contact_z - true_any_contact_z))
                err_touch_toe.append(abs(pred_touch_toe - true_touch_toe))
                err_touch_heel.append(abs(pred_touch_heel - true_touch_heel))
        records.append(record)

    summary = {
        "agent_checkpoint": args.agent_checkpoint,
        "trace_dir": args.trace_dir,
        "num_samples": int(len(records)),
        "horizon": int(args.horizon),
        "samples_per_trace": int(args.samples_per_trace),
        "state_name_filter": list(args.state_name_filter or []),
        "require_trace_landing_within": int(args.require_trace_landing_within),
        "touch_threshold": float(args.touch_threshold),
        "ground_eps": float(args.ground_eps),
        "events": {
            event: summarise_event(records, event, int(args.horizon))
            for event in event_names
        },
        "near_true_ground_onset_abs_error": {
            "vx": _error_stats(err_vx),
            "vz": _error_stats(err_vz),
            "omega": _error_stats(err_omega),
            "any_contact_z": _error_stats(err_any_contact_z),
            "touch_toe": _error_stats(err_touch_toe),
            "touch_heel": _error_stats(err_touch_heel),
        },
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (out_dir / "records.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
