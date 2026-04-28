#!/usr/bin/env python3
"""Evaluate Hopper action ranking quality on restorable official-eval trace states."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
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

os.environ.setdefault("MUJOCO_GL", "glfw")

from common import math as td_math
from common.sd_world_model import SDWorldModel
from scripts.analyze_hopper_sd_action_sensitivity import TraceState, _row_to_trace_state, iter_trace_paths
from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel


def make_env():
    from dm_control import suite
    from dm_control.suite.wrappers import action_scale

    env = suite.load("hopper", "hop", task_kwargs={"random": 0}, visualize_reward=False)
    return action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)


def _trace_state_name(row: dict[str, str]) -> str:
    sid = int(float(row["dominant_state_id"]))
    return DMControlHopperSDModel.STATE_NAMES[sid]


def _trace_row_grounded(row: dict[str, str]) -> bool:
    if "any_contact_z" in row:
        return float(row["any_contact_z"]) <= DMControlHopperSDModel.STATE_GROUND_EPS
    if "foot_grounded" in row:
        return float(row["foot_grounded"]) >= 0.5
    return False


def load_trace_states_filtered(
    trace_paths: Iterable[str],
    samples_per_trace: int,
    min_step: int,
    max_step: int | None,
    exclude_done: bool,
    state_name_filter: Sequence[str] | None,
    require_trace_landing_within: int | None,
) -> List[TraceState]:
    states: List[TraceState] = []
    allowed_state_names = set(state_name_filter or [])
    for trace_path in trace_paths:
        with open(trace_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        candidate_rows: List[dict[str, str]] = []
        for idx, row in enumerate(rows):
            step = int(float(row["step"]))
            if exclude_done and int(float(row.get("done", 0) or 0)) != 0:
                continue
            if step < min_step:
                continue
            if max_step is not None and step > max_step:
                continue
            if allowed_state_names and _trace_state_name(row) not in allowed_state_names:
                continue
            if require_trace_landing_within is not None:
                if float(row.get("airborne_true", 0.0)) < 0.5:
                    continue
                future = rows[idx + 1 : idx + 1 + int(require_trace_landing_within)]
                if not future:
                    continue
                if not any(_trace_row_grounded(fut) for fut in future):
                    continue
            candidate_rows.append(row)
        if not candidate_rows:
            continue
        keep = min(samples_per_trace, len(candidate_rows))
        idxs = np.linspace(0, len(candidate_rows) - 1, num=keep, dtype=int)
        for idx in idxs.tolist():
            states.append(_row_to_trace_state(trace_path, candidate_rows[idx]))
    if not states:
        raise RuntimeError("No trace states remained after filtering by step/done constraints.")
    return states


@dataclass
class MiniCfg:
    world_model_kind: str = "sd_hopper"
    obs: str = "state"
    multitask: bool = False
    task: str = "hopper-hop"
    action_dim: int = DMControlHopperSDModel.ACTION_DIM
    obs_shape: Dict[str, tuple] = None
    mlp_dim: int = 256
    num_bins: int = 101
    num_q: int = 3
    dropout: float = 0.01
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    episodic: bool = False
    sd_struct_dim: int = DMControlHopperSDModel.STATE_DIM
    sd_latent_dim: int = DMControlHopperSDModel.STATE_DIM
    latent_dim: int = DMControlHopperSDModel.STATE_DIM
    sd_reward_mode: str = "learned"
    sd_env_reward_scale: float = 1.0
    sd_reward_hybrid_alpha: float = 0.0
    sd_reward_curriculum_start: int = 0
    sd_reward_curriculum_end: int = 0
    sd_reward_curriculum_alpha: float = 0.0
    sd_action_prior_enabled: bool = False
    sd_dt: float = 0.02
    sd_hidden_dim: int = 256
    sd_hop_speed: float = 2.0
    sd_shaping_scale: float = 0.0
    sd_dense_pose_delta_weight: float = 0.0
    sd_dense_pose_abs_weight: float = 0.0
    sd_dense_pose_milestone_scale: float = 0.0
    sd_dense_pose_gamma: float = 0.99
    sd_recover_penalty_scale: float = 0.0
    sd_stage_mode: str = "full"
    sd_use_transition_residual: bool = True
    sd_transition_residual_scale: float = 4.0
    vmin: float = -10.0
    vmax: float = 10.0
    bin_size: float = 0.2
    discount_min: float = 0.95
    discount_max: float = 0.995
    task_dim: int = 0
    tasks: List[str] = None

    def __post_init__(self):
        if self.obs_shape is None:
            self.obs_shape = {"state": (15,)}
        if self.tasks is None:
            self.tasks = [self.task]

    def get(self, key, default=None):
        return getattr(self, key, default)


def build_world_model(args) -> SDWorldModel:
    cfg = MiniCfg(
        mlp_dim=int(args.mlp_dim),
        num_bins=int(args.num_bins),
        num_q=int(args.num_q),
        dropout=float(args.dropout),
        log_std_min=float(args.log_std_min),
        log_std_max=float(args.log_std_max),
        sd_struct_dim=int(args.sd_struct_dim),
        sd_latent_dim=int(args.sd_latent_dim),
        latent_dim=int(args.sd_latent_dim),
        sd_hidden_dim=int(args.sd_hidden_dim),
        sd_hop_speed=float(args.sd_hop_speed),
        sd_transition_residual_scale=float(args.sd_transition_residual_scale),
        discount_max=float(args.discount),
    )
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    model = SDWorldModel(cfg)
    payload = torch.load(args.agent_checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload["model"] if "model" in payload else payload
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _restore_env_state(env, sample: TraceState) -> None:
    _ = env.reset()
    physics = env.physics
    physics.data.qpos[:] = sample.qpos
    physics.data.qvel[:] = sample.qvel
    physics.forward()


def _env_step_decision(env, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
    total_reward = 0.0
    timestep = None
    action64 = action.astype(np.float64)
    for _ in range(DMControlHopperSDModel.ENV_ACTION_REPEAT):
        timestep = env.step(action64)
        total_reward += 0.0 if timestep.reward is None else float(timestep.reward)
        if timestep.last():
            break
    assert timestep is not None
    next_raw = DMControlHopperSDModel.flatten_obs(timestep.observation).astype(np.float32)
    return next_raw, total_reward, bool(timestep.last())


def _policy_mean(model: SDWorldModel, z: torch.Tensor) -> torch.Tensor:
    _action, info = model.pi(z, None)
    return info["mean"]


def _reward_scalar(model: SDWorldModel, z: torch.Tensor, a: torch.Tensor, next_z: torch.Tensor) -> torch.Tensor:
    logits = model.reward(z, a, None, next_z=next_z)
    return td_math.two_hot_inv(logits, model.cfg)


def _predict_model_score(
    model: SDWorldModel,
    raw_obs: np.ndarray,
    first_action: np.ndarray,
    horizon: int,
    discount: float,
) -> Dict[str, float]:
    obs_t = torch.from_numpy(raw_obs).float().unsqueeze(0)
    z = model.encode(obs_t, None)
    a = torch.from_numpy(first_action).float().unsqueeze(0)
    reward_sum = 0.0
    cur_discount = 1.0
    with torch.no_grad():
        for t in range(horizon):
            next_z = model.next(z, a, None)
            reward = float(_reward_scalar(model, z, a, next_z).item())
            reward_sum += cur_discount * reward
            z = next_z
            cur_discount *= discount
            if t < horizon - 1:
                a = _policy_mean(model, z)
        tail_a = _policy_mean(model, z)
        tail_q = float(model.Q(z, tail_a, None, return_type="avg").item())
    return {
        "pred_reward_sum": reward_sum,
        "pred_tail_q": tail_q,
        "pred_bootstrap_score": reward_sum + cur_discount * tail_q,
    }


def _true_env_score(
    env,
    model: SDWorldModel,
    sample: TraceState,
    first_action: np.ndarray,
    horizon: int,
    discount: float,
) -> Dict[str, float]:
    _restore_env_state(env, sample)
    obs = sample.raw_obs.copy()
    action = first_action.astype(np.float32, copy=True)
    reward_sum = 0.0
    cur_discount = 1.0
    done = False
    with torch.no_grad():
        for t in range(horizon):
            obs, reward, done = _env_step_decision(env, action)
            reward_sum += cur_discount * reward
            cur_discount *= discount
            if done:
                break
            if t < horizon - 1:
                z_true = model.encode(torch.from_numpy(obs).float().unsqueeze(0), None)
                action = _policy_mean(model, z_true).squeeze(0).cpu().numpy().astype(np.float32)
        if done:
            tail_q = 0.0
        else:
            z_true = model.encode(torch.from_numpy(obs).float().unsqueeze(0), None)
            tail_a = _policy_mean(model, z_true)
            tail_q = float(model.Q(z_true, tail_a, None, return_type="avg").item())
    return {
        "true_env_return": reward_sum,
        "true_tail_q": tail_q,
        "true_bootstrap_score": reward_sum + (0.0 if done else cur_discount * tail_q),
        "done_within_horizon": float(done),
    }


def _true_env_long_return(
    env,
    model: SDWorldModel,
    sample: TraceState,
    first_action: np.ndarray,
    horizon: int,
    discount: float,
) -> float:
    _restore_env_state(env, sample)
    obs = sample.raw_obs.copy()
    action = first_action.astype(np.float32, copy=True)
    reward_sum = 0.0
    cur_discount = 1.0
    with torch.no_grad():
        for t in range(horizon):
            obs, reward, done = _env_step_decision(env, action)
            reward_sum += cur_discount * reward
            if done:
                break
            cur_discount *= discount
            if t < horizon - 1:
                z_true = model.encode(torch.from_numpy(obs).float().unsqueeze(0), None)
                action = _policy_mean(model, z_true).squeeze(0).cpu().numpy().astype(np.float32)
    return float(reward_sum)


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = _rankdata(x)
    ry = _rankdata(y)
    if np.std(rx) < 1e-8 or np.std(ry) < 1e-8:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _best_case_rank(scores: np.ndarray, idx: int, eps: float = 1e-10) -> int:
    return int(1 + np.sum(scores > (scores[idx] + eps)))


def _candidate_actions(
    model: SDWorldModel,
    sample: TraceState,
    rng: np.random.Generator,
    num_random: int,
    random_std: float,
    basis_amp: float,
) -> List[tuple[str, np.ndarray]]:
    with torch.no_grad():
        z = model.encode(torch.from_numpy(sample.raw_obs).float().unsqueeze(0), None)
        policy_mean = _policy_mean(model, z).squeeze(0).cpu().numpy().astype(np.float32)
    out: List[tuple[str, np.ndarray]] = [("policy_mean", policy_mean.copy())]
    out.append(("zero", np.zeros((DMControlHopperSDModel.ACTION_DIM,), dtype=np.float32)))
    for dim, name in enumerate(("waist", "hip", "knee", "ankle")):
        for sign, sign_name in ((-1.0, "neg"), (1.0, "pos")):
            action = policy_mean.copy()
            action[dim] = float(np.clip(sign * basis_amp, -1.0, 1.0))
            out.append((f"{name}_{sign_name}_{basis_amp:.2f}", action.astype(np.float32)))
    for idx in range(num_random):
        action = np.clip(policy_mean + rng.normal(0.0, random_std, size=policy_mean.shape), -1.0, 1.0).astype(np.float32)
        out.append((f"random_{idx:02d}", action))
    return out


def _state_name(sample: TraceState) -> str:
    return f"{Path(sample.source_csv).stem}:step{sample.step}"


def _base_state_info(model: SDWorldModel, sample: TraceState) -> Dict[str, object]:
    dynamics = model._dynamics
    with torch.no_grad():
        z = model.encode(torch.from_numpy(sample.raw_obs).float().unsqueeze(0), None)
        z_struct = z[:, : dynamics.STATE_DIM]
        feat = dynamics._state_reward_features(z_struct)
        sid = int(dynamics._dominant_state_ids(feat)[0].item())
    return {
        "base_state_id": sid,
        "base_state_name": dynamics.STATE_NAMES[sid],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hopper SD action ranking diagnostic on official eval trace states.")
    parser.add_argument("--agent-checkpoint", required=True, help="TD-MPC2 agent checkpoint (.pt) containing reward/Q/pi heads.")
    parser.add_argument("--trace-csv", action="append", default=[], help="Path to a specific official_eval episode csv.")
    parser.add_argument("--trace-dir", action="append", default=[], help="Directory containing official_eval episode csvs.")
    parser.add_argument("--samples-per-trace", type=int, default=2)
    parser.add_argument("--min-step", type=int, default=0)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--exclude-done", action="store_true")
    parser.add_argument("--state-name-filter", nargs="+", default=None, choices=list(DMControlHopperSDModel.STATE_NAMES))
    parser.add_argument("--require-trace-landing-within", type=int, default=None)
    parser.add_argument("--num-random-actions", type=int, default=12)
    parser.add_argument("--random-std", type=float, default=0.35)
    parser.add_argument("--basis-amp", type=float, default=0.75)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--true-env-horizon", type=int, default=10)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mlp-dim", type=int, default=256)
    parser.add_argument("--num-q", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--num-bins", type=int, default=101)
    parser.add_argument("--log-std-min", type=float, default=-10.0)
    parser.add_argument("--log-std-max", type=float, default=2.0)
    parser.add_argument("--sd-struct-dim", type=int, default=17)
    parser.add_argument("--sd-latent-dim", type=int, default=17)
    parser.add_argument("--sd-hidden-dim", type=int, default=256)
    parser.add_argument("--sd-hop-speed", type=float, default=2.0)
    parser.add_argument("--sd-transition-residual-scale", type=float, default=4.0)
    args = parser.parse_args()

    trace_paths: List[str] = []
    for d in args.trace_dir:
        trace_paths.extend(iter_trace_paths([], d))
    trace_paths.extend(args.trace_csv)
    trace_paths = sorted(dict.fromkeys(trace_paths))
    if not trace_paths:
        raise RuntimeError("At least one --trace-dir or --trace-csv is required.")

    model = build_world_model(args)
    env = make_env()
    samples = load_trace_states_filtered(
        trace_paths=trace_paths,
        samples_per_trace=int(args.samples_per_trace),
        min_step=int(args.min_step),
        max_step=None if args.max_step is None else int(args.max_step),
        exclude_done=bool(args.exclude_done),
        state_name_filter=args.state_name_filter,
        require_trace_landing_within=args.require_trace_landing_within,
    )
    rng = np.random.default_rng(args.seed)

    details: List[Dict[str, object]] = []
    per_state_rows: List[Dict[str, object]] = []

    for sample_idx, sample in enumerate(samples):
        candidates = _candidate_actions(
            model=model,
            sample=sample,
            rng=rng,
            num_random=int(args.num_random_actions),
            random_std=float(args.random_std),
            basis_amp=float(args.basis_amp),
        )
        state_info = _base_state_info(model, sample)
        pred_scores = []
        true_boot_scores = []
        true_env_scores = []
        candidate_names = []

        for cand_idx, (cand_name, action) in enumerate(candidates):
            pred = _predict_model_score(model, sample.raw_obs, action, int(args.horizon), float(args.discount))
            true = _true_env_score(env, model, sample, action, int(args.horizon), float(args.discount))
            true_long = _true_env_long_return(
                env=env,
                model=model,
                sample=sample,
                first_action=action,
                horizon=int(args.true_env_horizon),
                discount=float(args.discount),
            )
            row = {
                "sample_idx": sample_idx,
                "state_label": _state_name(sample),
                "source_csv": sample.source_csv,
                "episode": sample.episode,
                "step": sample.step,
                "candidate_idx": cand_idx,
                "candidate_name": cand_name,
                "action_0": float(action[0]),
                "action_1": float(action[1]),
                "action_2": float(action[2]),
                "action_3": float(action[3]),
                **state_info,
                **pred,
                **true,
                "true_long_env_return": true_long,
            }
            details.append(row)
            pred_scores.append(pred["pred_bootstrap_score"])
            true_boot_scores.append(true["true_bootstrap_score"])
            true_env_scores.append(true["true_env_return"])
            candidate_names.append(cand_name)
        true_long_np = np.asarray([float(r["true_long_env_return"]) for r in details[-len(candidates):]], dtype=np.float64)

        pred_scores_np = np.asarray(pred_scores, dtype=np.float64)
        true_boot_np = np.asarray(true_boot_scores, dtype=np.float64)
        true_env_np = np.asarray(true_env_scores, dtype=np.float64)

        pred_best = int(np.argmax(pred_scores_np))
        true_best = int(np.argmax(true_boot_np))
        policy_idx = candidate_names.index("policy_mean")
        pred_rank_of_true_best = _best_case_rank(pred_scores_np, true_best)
        true_rank_of_pred_best = _best_case_rank(true_boot_np, pred_best)
        true_boot_std = float(np.std(true_boot_np))
        pred_boot_std = float(np.std(pred_scores_np))
        per_state_rows.append(
            {
                "sample_idx": sample_idx,
                "state_label": _state_name(sample),
                "source_csv": sample.source_csv,
                "episode": sample.episode,
                "step": sample.step,
                **state_info,
                "num_candidates": len(candidates),
                "spearman_pred_vs_true_boot": _spearman(pred_scores_np, true_boot_np),
                "spearman_pred_vs_true_env": _spearman(pred_scores_np, true_env_np),
                "spearman_pred_vs_true_long_env": _spearman(pred_scores_np, true_long_np),
                "pred_rank_of_true_best": pred_rank_of_true_best,
                "true_rank_of_pred_best": true_rank_of_pred_best,
                "top1_hit": float(pred_rank_of_true_best <= 1),
                "top3_contains_true_top1": float(pred_rank_of_true_best <= 3),
                "pred_best_name": candidate_names[pred_best],
                "true_best_name": candidate_names[true_best],
                "true_boot_std": true_boot_std,
                "pred_boot_std": pred_boot_std,
                "nondegenerate_true_boot": float(true_boot_std > 1e-6),
                "pred_best_true_boot": float(true_boot_np[pred_best]),
                "true_best_true_boot": float(true_boot_np[true_best]),
                "policy_true_boot": float(true_boot_np[policy_idx]),
                "pred_best_true_env": float(true_env_np[pred_best]),
                "true_best_true_env": float(true_env_np[true_best]),
                "pred_best_true_long_env": float(true_long_np[pred_best]),
                "true_best_true_long_env": float(true_long_np[true_best]),
                "any_nonzero_env_reward": float(np.any(np.abs(true_env_np) > 1e-8)),
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_csv = out_dir / "action_ranking_details.csv"
    with detail_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0].keys()))
        writer.writeheader()
        writer.writerows(details)

    state_csv = out_dir / "action_ranking_per_state.csv"
    with state_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_state_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_state_rows)

    def _mean(key: str, rows: Sequence[Dict[str, object]] | None = None) -> float:
        rows = per_state_rows if rows is None else rows
        vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        return float(finite.mean()) if finite.size else float("nan")

    def _nanmean_list(values: Sequence[float]) -> float:
        vals = np.asarray(list(values), dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        return float(finite.mean()) if finite.size else float("nan")

    informative_rows = [r for r in per_state_rows if float(r["nondegenerate_true_boot"]) >= 0.5]

    by_state: Dict[str, Dict[str, float]] = {}
    for state_name in DMControlHopperSDModel.STATE_NAMES:
        rows = [r for r in per_state_rows if r["base_state_name"] == state_name]
        if not rows:
            continue
        by_state[state_name] = {
            "count": int(len(rows)),
            "spearman_pred_vs_true_boot_mean": _nanmean_list([float(r["spearman_pred_vs_true_boot"]) for r in rows]),
            "spearman_pred_vs_true_long_env_mean": _nanmean_list([float(r["spearman_pred_vs_true_long_env"]) for r in rows]),
            "top1_hit_mean": float(np.mean([float(r["top1_hit"]) for r in rows])),
            "top3_contains_true_top1_mean": float(np.mean([float(r["top3_contains_true_top1"]) for r in rows])),
            "pred_best_true_boot_mean": float(np.mean([float(r["pred_best_true_boot"]) for r in rows])),
            "true_best_true_boot_mean": float(np.mean([float(r["true_best_true_boot"]) for r in rows])),
        }

    summary = {
        "agent_checkpoint": str(args.agent_checkpoint),
        "trace_paths": trace_paths,
        "num_samples": len(samples),
        "num_candidates_per_state": int(len(details) / max(len(samples), 1)),
        "horizon": int(args.horizon),
        "true_env_horizon": int(args.true_env_horizon),
        "discount": float(args.discount),
        "detail_csv": str(detail_csv),
        "per_state_csv": str(state_csv),
        "aggregate": {
            "informative_state_count": int(len(informative_rows)),
            "spearman_pred_vs_true_boot_mean": _mean("spearman_pred_vs_true_boot"),
            "spearman_pred_vs_true_env_mean": _mean("spearman_pred_vs_true_env"),
            "spearman_pred_vs_true_long_env_mean": _mean("spearman_pred_vs_true_long_env"),
            "top1_hit_mean": _mean("top1_hit"),
            "top3_contains_true_top1_mean": _mean("top3_contains_true_top1"),
            "pred_best_true_boot_mean": _mean("pred_best_true_boot"),
            "true_best_true_boot_mean": _mean("true_best_true_boot"),
            "policy_true_boot_mean": _mean("policy_true_boot"),
            "pred_best_true_env_mean": _mean("pred_best_true_env"),
            "true_best_true_env_mean": _mean("true_best_true_env"),
            "pred_best_true_long_env_mean": _mean("pred_best_true_long_env"),
            "true_best_true_long_env_mean": _mean("true_best_true_long_env"),
            "any_nonzero_env_reward_frac": _mean("any_nonzero_env_reward"),
            "informative_spearman_pred_vs_true_boot_mean": _mean("spearman_pred_vs_true_boot", informative_rows),
            "informative_spearman_pred_vs_true_env_mean": _mean("spearman_pred_vs_true_env", informative_rows),
            "informative_spearman_pred_vs_true_long_env_mean": _mean("spearman_pred_vs_true_long_env", informative_rows),
            "informative_top1_hit_mean": _mean("top1_hit", informative_rows),
            "informative_top3_contains_true_top1_mean": _mean("top3_contains_true_top1", informative_rows),
            "informative_pred_best_true_boot_mean": _mean("pred_best_true_boot", informative_rows),
            "informative_true_best_true_boot_mean": _mean("true_best_true_boot", informative_rows),
            "informative_pred_best_true_long_env_mean": _mean("pred_best_true_long_env", informative_rows),
            "informative_true_best_true_long_env_mean": _mean("true_best_true_long_env", informative_rows),
        },
        "by_base_state": by_state,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
