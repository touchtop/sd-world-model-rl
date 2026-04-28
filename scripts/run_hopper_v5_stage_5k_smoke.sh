#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:?usage: bash scripts/run_hopper_v5_stage_5k_smoke.sh <gpu_id> <exp_name> <seed> <stage_mode> <reset_pool_csv>}"
EXP_NAME="${2:?usage: bash scripts/run_hopper_v5_stage_5k_smoke.sh <gpu_id> <exp_name> <seed> <stage_mode> <reset_pool_csv>}"
SEED="${3:?usage: bash scripts/run_hopper_v5_stage_5k_smoke.sh <gpu_id> <exp_name> <seed> <stage_mode> <reset_pool_csv>}"
STAGE_MODE="${4:?usage: bash scripts/run_hopper_v5_stage_5k_smoke.sh <gpu_id> <exp_name> <seed> <stage_mode> <reset_pool_csv>}"
RESET_POOL_CSV="${5:?usage: bash scripts/run_hopper_v5_stage_5k_smoke.sh <gpu_id> <exp_name> <seed> <stage_mode> <reset_pool_csv>}"
RESET_POOL_CSV_ABS="$(cd "$(dirname "$RESET_POOL_CSV")" && pwd)/$(basename "$RESET_POOL_CSV")"

case "$STAGE_MODE" in
  landing|ground|rise|prone_recover) ;;
  *)
    echo "invalid stage_mode: $STAGE_MODE (expected: landing|ground|rise|prone_recover)" >&2
    exit 2
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-/home/anaconda3/envs/sdmcts_p100/bin/python}"

export PYTHONPATH="$ROOT/external/tdmpc2/tdmpc2:$ROOT:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

cat <<EOF
[hopper-v5-stage-5k-smoke]
root=$ROOT
python=$PYTHON_BIN
gpu=$GPU_ID
seed=$SEED
exp_name=$EXP_NAME
stage_mode=$STAGE_MODE
reset_pool_csv=$RESET_POOL_CSV_ABS
steps=5000
eval_freq=5000
checkpoint_freq=5000
eval_episodes=10
sd_struct_dim=17
sd_latent_dim=512
sd_reward_mode=hybrid
sd_env_reward_scale=0.0
sd_reward_hybrid_alpha=1.0

Diagnostic intent:
- isolate a single Hopper stage
- reset every episode from the selected stage pool
- train reward head on shaping only
EOF

exec "$PYTHON_BIN" -u "$ROOT/external/tdmpc2/tdmpc2/train.py" \
  task=hopper-hop \
  world_model_kind=sd_hopper \
  model_size=null \
  obs=state \
  steps=5000 \
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
  sd_latent_dim=512 \
  num_samples=128 \
  num_elites=16 \
  num_pi_trajs=16 \
  horizon=3 \
  iterations=3 \
  sd_stage_mode="$STAGE_MODE" \
  hopper_stage_reset_pool_csv="$RESET_POOL_CSV_ABS" \
  sd_reward_mode=hybrid \
  sd_env_reward_scale=0.0 \
  sd_reward_hybrid_alpha=1.0 \
  +sd_shaping_scale=1.0 \
  sd_max_resid_scale=5.0 \
  sd_height_loss_coef=0.05
