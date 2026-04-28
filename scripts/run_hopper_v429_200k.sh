#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:?usage: bash scripts/run_hopper_v429_200k.sh <gpu_id> [exp_name] [seed]}"
EXP_NAME="${2:-sd_hopper_200000_v429_seed}"
SEED="${3:-1}"

PYTHON_BIN="${PYTHON_BIN:-/home/anaconda3/envs/sdmcts_p100/bin/python}"

export PYTHONPATH="$ROOT/external/tdmpc2/tdmpc2:$ROOT:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

cat <<EOF
[hopper-v429-200k]
root=$ROOT
python=$PYTHON_BIN
gpu=$GPU_ID
seed=$SEED
exp_name=$EXP_NAME
steps=200000
eval_freq=25000
checkpoint_freq=25000
eval_episodes=10

Fixed invariants:
- Hopper episode_length=500
- TD-MPC2 sets seed_steps=max(1000, 5 * episode_length)=2500
- No official eval before seed collection ends
- Official eval automatically exports traces and saves resumable full state
EOF

exec "$PYTHON_BIN" -u "$ROOT/external/tdmpc2/tdmpc2/train.py" \
  task=hopper-hop \
  world_model_kind=sd_hopper \
  model_size=null \
  obs=state \
  steps=200000 \
  seed="$SEED" \
  exp_name="$EXP_NAME" \
  enable_wandb=false \
  save_video=false \
  save_agent=true \
  save_csv=true \
  save_full_state=true \
  eval_episodes=10 \
  eval_freq=25000 \
  checkpoint_freq=25000 \
  compile=false \
  batch_size=128 \
  lr=3e-4 \
  dropout=0.01 \
  mlp_dim=256 \
  num_q=3 \
  sd_struct_dim=9 \
  sd_latent_dim=512 \
  num_samples=128 \
  num_elites=16 \
  num_pi_trajs=16 \
  horizon=3 \
  iterations=3 \
  sd_reward_mode=curriculum \
  sd_reward_curriculum_alpha=0.5 \
  sd_reward_curriculum_alpha_min=0.25 \
  sd_reward_curriculum_start=10000 \
  sd_reward_curriculum_end=30000 \
  sd_max_resid_scale=5.0 \
  sd_height_loss_coef=0.05
