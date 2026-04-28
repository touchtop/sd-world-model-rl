#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:?usage: bash scripts/run_hopper_v5_10k_pureenv_sdinit.sh <gpu_id> [exp_name] [seed]}"
EXP_NAME="${2:-sd_hopper_10k_pureenv_sdinit_smoke}"
SEED="${3:-3}"

PYTHON_BIN="${PYTHON_BIN:-/home/anaconda3/envs/sdmcts_p100/bin/python}"
SD_INIT_CKPT="${SD_INIT_CKPT:-$ROOT/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_shortloss010_cpu/hopper_sd_supervised.pt}"

export PYTHONPATH="$ROOT/external/tdmpc2/tdmpc2:$ROOT:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

cat <<EOF
[hopper-v5-10k-pureenv-sdinit]
root=$ROOT
python=$PYTHON_BIN
gpu=$GPU_ID
seed=$SEED
exp_name=$EXP_NAME
steps=10000
eval_freq=5000
checkpoint_freq=5000
eval_episodes=10
sd_init_ckpt=$SD_INIT_CKPT

Intent:
- keep TD-MPC2 reward learning on environment reward only
- disable shaping / curriculum mixing / action prior
- keep sd_latent_dim=17 so planning uses explicit SD state only
- enable internal transition residual inside DMControlHopperSDModel
- initialize SD dynamics from the best current supervised short-horizon checkpoint
EOF

test -f "$SD_INIT_CKPT"

exec "$PYTHON_BIN" -u "$ROOT/external/tdmpc2/tdmpc2/train.py" \
  task=hopper-hop \
  world_model_kind=sd_hopper \
  model_size=null \
  obs=state \
  steps=10000 \
  seed="$SEED" \
  exp_name="$EXP_NAME" \
  enable_wandb=false \
  save_video=false \
  save_agent=true \
  save_csv=true \
  save_full_state=true \
  save_eval_traces=true \
  eval_episodes=10 \
  eval_freq=5000 \
  checkpoint_freq=5000 \
  compile=false \
  batch_size=128 \
  lr=3e-4 \
  dropout=0.01 \
  mlp_dim=256 \
  num_q=3 \
  sd_struct_dim=17 \
  sd_latent_dim=17 \
  sd_use_transition_residual=true \
  sd_transition_residual_scale=4.0 \
  sd_reward_mode=learned \
  sd_env_reward_scale=1.0 \
  sd_action_prior_enabled=false \
  sd_stage_mode=full \
  sd_height_loss_coef=0.0 \
  num_samples=128 \
  num_elites=16 \
  num_pi_trajs=16 \
  horizon=3 \
  iterations=3 \
  sd_supervised_init_from="$SD_INIT_CKPT"
