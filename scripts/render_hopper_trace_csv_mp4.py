import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import imageio
import numpy as np
from dm_control import suite


QPOS_KEYS = [
    "qpos_rootx",
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_csv", type=Path, required=True)
    parser.add_argument("--out_mp4", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--camera_id", type=int, default=0)
    return parser.parse_args()


def load_rows(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_to_state(row):
    qpos = np.array([float(row[k]) for k in QPOS_KEYS], dtype=np.float64)
    qvel = np.array([float(row[k]) for k in QVEL_KEYS], dtype=np.float64)
    return qpos, qvel


def main():
    args = parse_args()
    rows = load_rows(args.trace_csv)
    if not rows:
        raise ValueError(f"empty trace csv: {args.trace_csv}")

    env = suite.load("hopper", "hop")
    physics = env.physics
    frames = []

    for row in rows:
        qpos, qvel = row_to_state(row)
        physics.data.qpos[:] = qpos
        physics.data.qvel[:] = qvel
        physics.forward()
        frames.append(physics.render(height=args.height, width=args.width, camera_id=args.camera_id))

    args.out_mp4.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out_mp4, frames, fps=args.fps)
    print(f"trace_csv={args.trace_csv}")
    print(f"frames={len(frames)}")
    print(f"out_mp4={args.out_mp4}")


if __name__ == "__main__":
    main()
