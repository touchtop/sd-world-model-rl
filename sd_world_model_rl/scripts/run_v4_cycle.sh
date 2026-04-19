#!/usr/bin/env bash
set -euo pipefail

# Train one v4 Hopper smoke, then immediately run the closed-loop diagnosis
# on its final checkpoint. This keeps one iteration self-contained: logs,
# videos, CSV summaries, and diagnosis report all land under fresh paths.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-3}"
STEPS="${STEPS:-3000}"
EVAL_FREQ="${EVAL_FREQ:-1000}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
MLP_DIM="${MLP_DIM:-256}"
SD_LATENT_DIM="${SD_LATENT_DIM:-32}"
NUM_Q="${NUM_Q:-3}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-3e-4}"
DROPOUT="${DROPOUT:-0.01}"
NUM_SAMPLES="${NUM_SAMPLES:-128}"
NUM_ELITES="${NUM_ELITES:-16}"
NUM_PI_TRAJS="${NUM_PI_TRAJS:-16}"
HORIZON="${HORIZON:-3}"
ITERATIONS="${ITERATIONS:-3}"
EXP_LABEL="${EXP_LABEL:-v4_cycle}"
EXP_NAME="sd_hopper_${STEPS}_${EXP_LABEL}_${DATE_TAG}"
TRAIN_LOG_DIR="results/${EXP_LABEL}"
TRAIN_STDOUT="${TRAIN_LOG_DIR}/${EXP_NAME}.log"
DIAG_OUT_DIR="results/closed_loop/${EXP_NAME}_quick3"

mkdir -p "$TRAIN_LOG_DIR"

env MUJOCO_GL=glfw CUDA_VISIBLE_DEVICES="$GPU_ID" \
  /home/anaconda3/envs/sdmcts_p100/bin/python -u external/tdmpc2/tdmpc2/train.py \
  task=hopper-hop world_model_kind=sd_hopper model_size=null obs=state \
  steps="$STEPS" seed="$SEED" exp_name="$EXP_NAME" \
  enable_wandb=false save_video=false save_agent=true save_csv=true \
  eval_episodes="$EVAL_EPISODES" eval_freq="$EVAL_FREQ" compile=false \
  batch_size="$BATCH_SIZE" lr="$LR" dropout="$DROPOUT" \
  mlp_dim="$MLP_DIM" sd_latent_dim="$SD_LATENT_DIM" num_q="$NUM_Q" \
  num_samples="$NUM_SAMPLES" num_elites="$NUM_ELITES" num_pi_trajs="$NUM_PI_TRAJS" \
  horizon="$HORIZON" iterations="$ITERATIONS" \
  sd_reward_mode=curriculum sd_reward_curriculum_alpha=0.5 \
  sd_reward_curriculum_alpha_min=0.25 sd_reward_curriculum_start=10000 \
  sd_reward_curriculum_end=30000 sd_max_resid_scale=5.0 \
  sd_height_loss_coef=0.05 sd_use_struct_residual=false \
  > "$TRAIN_STDOUT" 2>&1

xvfb-run -a -s "-screen 0 1400x900x24" \
  /home/anaconda3/envs/sdmcts_p100/bin/python scripts/v4_hopper_closed_loop.py \
  --checkpoint "logs/hopper-hop/${SEED}/${EXP_NAME}/models/final.pt" \
  --out_dir "$DIAG_OUT_DIR" \
  --task hopper-hop --seed "$SEED" --episodes 3 --fps 15 \
  --render_failures 3 --render_successes 0 --single_pass true \
  --world_model_kind sd_hopper --mlp_dim "$MLP_DIM" --num_q "$NUM_Q" \
  --num_samples "$NUM_SAMPLES" --num_elites "$NUM_ELITES" --num_pi_trajs "$NUM_PI_TRAJS" \
  --horizon "$HORIZON" --iterations "$ITERATIONS" \
  --sd_reward_mode curriculum --sd_reward_curriculum_alpha 0.5 \
  --sd_reward_curriculum_alpha_min 0.25 --sd_reward_curriculum_start 10000 \
  --sd_reward_curriculum_end 30000 --sd_max_resid_scale 5.0 \
  --sd_height_loss_coef 0.05 --sd_use_struct_residual false

echo "train_log=${TRAIN_STDOUT}"
echo "diag_dir=${DIAG_OUT_DIR}"
