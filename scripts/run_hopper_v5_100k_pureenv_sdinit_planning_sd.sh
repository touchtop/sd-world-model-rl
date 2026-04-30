#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:?usage: bash scripts/run_hopper_v5_100k_pureenv_sdinit_planning_sd.sh <gpu_id> [exp_name] [seed]}"
EXP_NAME="${2:-sd_hopper_100k_pureenv_sdinit_planning_sd}"
SEED="${3:-1}"

PYTHON_BIN="${PYTHON_BIN:-/home/anaconda3/envs/sdmcts_p100/bin/python}"
SD_INIT_CKPT="${SD_INIT_CKPT:-$ROOT/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_shortloss010_cpu/hopper_sd_supervised.pt}"

export PYTHONPATH="$ROOT/external/tdmpc2/tdmpc2:$ROOT:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

cat <<EOF
[hopper-v5-100k-pureenv-sdinit-planning-sd]
root=$ROOT
python=$PYTHON_BIN
gpu=$GPU_ID
seed=$SEED
exp_name=$EXP_NAME
target_steps=100000
steps=100500
eval_freq=25000
checkpoint_freq=25000
eval_episodes=10
sd_init_ckpt=$SD_INIT_CKPT

Intent:
- keep pure environment reward and supervised SD initialization
- keep online SD supervision on the shared transition core
- add planning-aware sample weighting with TD-positive replay targets
- run 500 extra environment steps so the 100k eval trigger can reach an episode boundary
EOF

test -f "$SD_INIT_CKPT"

exec "$PYTHON_BIN" -u "$ROOT/external/tdmpc2/tdmpc2/train.py" \
  task=hopper-hop \
  world_model_kind=sd_hopper \
  model_size=null \
  obs=state \
  steps=100500 \
  seed="$SEED" \
  exp_name="$EXP_NAME" \
  enable_wandb=false \
  save_video=false \
  save_agent=true \
  save_csv=true \
  save_full_state=true \
  save_eval_traces=true \
  eval_episodes=10 \
  eval_freq=25000 \
  checkpoint_freq=25000 \
  compile=false \
  batch_size=128 \
  lr=3e-4 \
  dropout=0.01 \
  mlp_dim=256 \
  num_q=3 \
  sd_struct_dim=17 \
  sd_latent_dim=512 \
  sd_struct_loss_scale_mode=batch_std \
  sd_struct_loss_scale_floor=0.1 \
  sd_latent_consistency_struct_weight=1.0 \
  sd_latent_consistency_resid_weight=1.0 \
  sd_use_transition_residual=true \
  sd_transition_residual_scale=4.0 \
  sd_reward_mode=learned \
  sd_env_reward_scale=1.0 \
  sd_action_prior_enabled=false \
  sd_stage_mode=full \
  sd_height_loss_coef=0.0 \
  sd_online_supervision_enabled=true \
  sd_online_supervision_lr=3e-4 \
  sd_online_supervision_one_step_weight=1.0 \
  sd_online_supervision_rollout_weight=1.0 \
  sd_online_supervision_omega_weight=0.25 \
  sd_online_supervision_contact_weight=0.25 \
  sd_online_supervision_phase_weight=2.0 \
  sd_online_planning_weight_enabled=true \
  sd_online_planning_weight_mode=td_positive \
  sd_online_planning_weight_coef=1.0 \
  sd_online_planning_weight_max=5.0 \
  num_samples=128 \
  num_elites=16 \
  num_pi_trajs=16 \
  horizon=3 \
  iterations=3 \
  sd_supervised_init_from="$SD_INIT_CKPT"
