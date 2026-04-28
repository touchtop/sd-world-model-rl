import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import torch

from src.world_model.dmcontrol_hopper_sd_model import DMControlHopperSDModel


STATE_KEYS = [
    "falling_air",
    "grounded_rear",
    "grounded_prone",
    "grounded_upright",
    "grounded_prepared",
    "rising_air",
]

LATENT_KEYS = [
    "torso_z",
    "vx",
    "vz",
    "pitch",
    "pitch_rate",
    "touch_toe",
    "touch_heel",
    "z_min",
    "qpos_knee",
]

ACTION_KEYS = [f"action_{i}" for i in range(4)]
QPOS_KEYS = [
    "qpos_rootz",
    "qpos_rooty",
    "qpos_waist",
    "qpos_hip",
    "qpos_knee",
    "qpos_ankle",
]
QVEL_KEYS = [
    "qvel_rootx",
    "qvel_rootz",
    "qvel_rooty",
    "qvel_waist",
    "qvel_hip",
    "qvel_knee",
    "qvel_ankle",
]
CONTACT_KEYS = ["touch_toe", "touch_heel"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Validate Hopper six-state tags, transition shaping reward, and SD one-step consistency from saved trace CSVs."
    )
    p.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="label=glob_or_dir, e.g. v4_zero=logs/.../episode_csv or high=results/.../trace_probe_v428",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Optional shaping scale used only for reporting scaled shaping magnitude.",
    )
    return p.parse_args()


def resolve_paths(spec: str):
    label, raw_path = spec.split("=", 1)
    path = Path(raw_path)
    if path.is_dir():
        paths = sorted(path.glob("episode_*.csv"))
    else:
        paths = sorted(Path().glob(raw_path))
    if not paths:
        raise FileNotFoundError(f"No trace csv found for dataset {label}: {raw_path}")
    return label, paths


def load_rows(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f))


def dominant_state(row):
    values = {}
    for k in STATE_KEYS:
        values[k] = float(row.get(k, 0.0))
    return max(values, key=values.get)


def tensor_from_row(row, keys):
    return torch.tensor([[float(row[k]) for k in keys]], dtype=torch.float32)


def _contact_to_raw(contact: float) -> float:
    clipped = min(max(contact, 0.0), 1.0 - 1e-6)
    return float(-torch.log(torch.tensor(1.0 - clipped)).item())


def state_and_features_from_row(row, model):
    if all(k in row for k in QPOS_KEYS + QVEL_KEYS + CONTACT_KEYS):
        obs = torch.tensor([[
            float(row["qpos_rootz"]),
            float(row["qpos_rooty"]),
            float(row["qpos_waist"]),
            float(row["qpos_hip"]),
            float(row["qpos_knee"]),
            float(row["qpos_ankle"]),
            float(row["qvel_rootx"]),
            float(row["qvel_rootz"]),
            float(row["qvel_rooty"]),
            float(row["qvel_waist"]),
            float(row["qvel_hip"]),
            float(row["qvel_knee"]),
            float(row["qvel_ankle"]),
            _contact_to_raw(float(row["touch_toe"])),
            _contact_to_raw(float(row["touch_heel"])),
        ]], dtype=torch.float32)
        state = model.encode_observation(obs)
    else:
        state = tensor_from_row(row, LATENT_KEYS)
    feat = model._state_reward_features(state)
    values = {}
    for k in STATE_KEYS:
        values[k] = float(feat[k][0].item())
    return state, feat, values


def mean(values):
    return sum(values) / max(len(values), 1)


def analyze_dataset(label, paths, model, alpha):
    occ = Counter()
    transitions = Counter()
    reward_stats = defaultdict(list)
    state_reward_stats = {s: defaultdict(list) for s in STATE_KEYS}
    dim_error_stats = defaultdict(list)
    state_dim_error_stats = {s: defaultdict(list) for s in STATE_KEYS}
    per_trace_rows = []

    for path in paths:
        rows = load_rows(path)
        if len(rows) < 2:
            continue
        states = []
        features = []
        values_seq = []
        dom_seq = []
        for row in rows:
            state, feat, values = state_and_features_from_row(row, model)
            states.append(state)
            features.append(feat)
            values_seq.append(values)
            dom_seq.append(max(values, key=values.get))
        for s in dom_seq:
            occ[s] += 1
        for a, b in zip(dom_seq, dom_seq[1:]):
            transitions[(a, b)] += 1

        for i in range(len(rows) - 1):
            row = rows[i]
            state = states[i]
            action = tensor_from_row(row, ACTION_KEYS)
            next_state = states[i + 1]
            with torch.no_grad():
                pred_next = model(state, action)
                shaping = float(model.predict_shaping_reward(state, action, next_state).item())
                explicit_reward = float(model.predict_reward(state, action, next_state).item())
            env_reward = float(row["reward"])
            dom = dom_seq[i]

            reward_stats["env_reward"].append(env_reward)
            reward_stats["explicit_reward"].append(explicit_reward)
            reward_stats["shaping_raw"].append(shaping)
            reward_stats["shaping_scaled"].append(alpha * shaping)
            reward_stats["plan_reward"].append(explicit_reward + shaping)
            reward_stats["abs_gap_env_vs_explicit"].append(abs(env_reward - explicit_reward))

            state_reward_stats[dom]["env_reward"].append(env_reward)
            state_reward_stats[dom]["explicit_reward"].append(explicit_reward)
            state_reward_stats[dom]["shaping_raw"].append(shaping)
            state_reward_stats[dom]["shaping_scaled"].append(alpha * shaping)
            state_reward_stats[dom]["plan_reward"].append(explicit_reward + shaping)
            state_reward_stats[dom]["abs_gap_env_vs_explicit"].append(abs(env_reward - explicit_reward))

            abs_err = (pred_next - next_state).abs().squeeze(0)
            for idx, key in enumerate(LATENT_KEYS):
                val = float(abs_err[idx].item())
                dim_error_stats[key].append(val)
                state_dim_error_stats[dom][key].append(val)

        per_trace_rows.append(
            {
                "trace_csv": str(path),
                "reward_sum": sum(float(r["reward"]) for r in rows[:-1]),
                "dominant_state_counts": {s: dom_seq.count(s) for s in STATE_KEYS},
            }
        )

    total_occ = sum(occ.values())
    total_trans = sum(transitions.values())

    print(f"== {label} ==")
    print("state_occupancy")
    for s in STATE_KEYS:
        print(f"  {s}: frac={occ[s] / max(total_occ, 1):.4f} count={occ[s]}")

    print("top_transitions")
    for (a, b), c in transitions.most_common(12):
        print(f"  {a} -> {b}: frac={c / max(total_trans, 1):.4f} count={c}")

    print("reward_overall")
    for key in ["env_reward", "explicit_reward", "shaping_raw", "shaping_scaled", "plan_reward", "abs_gap_env_vs_explicit"]:
        vals = reward_stats[key]
        print(f"  {key}: mean={mean(vals):.6f} min={min(vals):.6f} max={max(vals):.6f}")

    print("reward_by_state")
    for s in STATE_KEYS:
        count = len(state_reward_stats[s]["env_reward"])
        if count == 0:
            continue
        print(f"  [{s}] count={count}")
        for key in ["env_reward", "explicit_reward", "shaping_raw", "shaping_scaled", "plan_reward", "abs_gap_env_vs_explicit"]:
            vals = state_reward_stats[s][key]
            print(f"    {key}: mean={mean(vals):.6f} min={min(vals):.6f} max={max(vals):.6f}")

    print("sd_one_step_mae_overall")
    for key in LATENT_KEYS:
        vals = dim_error_stats[key]
        print(f"  {key}: mean={mean(vals):.6f} max={max(vals):.6f}")

    print("sd_one_step_mae_by_state")
    for s in STATE_KEYS:
        counts = len(state_dim_error_stats[s][LATENT_KEYS[0]])
        if counts == 0:
            continue
        print(f"  [{s}] count={counts}")
        for key in LATENT_KEYS:
            vals = state_dim_error_stats[s][key]
            print(f"    {key}: mean={mean(vals):.6f} max={max(vals):.6f}")

    print("trace_examples")
    for row in per_trace_rows[:3]:
        print(f"  {row['trace_csv']}: reward_sum={row['reward_sum']:.6f} states={row['dominant_state_counts']}")
    print()


def main():
    args = parse_args()
    model = DMControlHopperSDModel().cpu()
    model.eval()
    for spec in args.dataset:
        label, paths = resolve_paths(spec)
        analyze_dataset(label, paths, model, args.alpha)


if __name__ == "__main__":
    main()
