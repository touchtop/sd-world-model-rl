#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:?usage: bash scripts/run_hopper_v6_5k_rule_prior_smoke.sh <gpu_id> [exp_name] [seed]}"
EXP_NAME="${2:-sd_hopper_v6_5k_rule_prior_smoke}"
SEED="${3:-1}"

PYTHON_BIN="${PYTHON_BIN:-/home/anaconda3/envs/sdmcts_p100/bin/python}"
SD_INIT_CKPT="${SD_INIT_CKPT:-$ROOT/results/hopper_sd_supervised/run_20260428_smoke_contactfix5_resid4_featctx_shortloss010_cpu/hopper_sd_supervised.pt}"
RULE_LIBRARY="${RULE_LIBRARY:-$ROOT/results/sd_retrospective_rule_induction/rules/library_v5.json}"
WORLD_MODEL_KIND="${WORLD_MODEL_KIND:-sd_hopper}"
SD_ONLINE_SUPERVISION_ENABLED="${SD_ONLINE_SUPERVISION_ENABLED:-true}"
RULE_PRIOR_ENABLED="${RULE_PRIOR_ENABLED:-false}"
RULE_PRIOR_WEIGHT="${RULE_PRIOR_WEIGHT:-0.05}"
RULE_PRIOR_OPPOSITE_PENALTY="${RULE_PRIOR_OPPOSITE_PENALTY:-0.02}"
RULE_PRIOR_MAX_BONUS="${RULE_PRIOR_MAX_BONUS:-0.10}"
RULE_PRIOR_MODE="${RULE_PRIOR_MODE:-prefer}"
RULE_PRIOR_PREFERRED_CAP="${RULE_PRIOR_PREFERRED_CAP:-0.70}"
RULE_PRIOR_CAP_PENALTY="${RULE_PRIOR_CAP_PENALTY:-0.02}"
RULE_PRIOR_ALLOWED_STATUSES="${RULE_PRIOR_ALLOWED_STATUSES:-replay_validated}"
RULE_PRIOR_STOCHASTIC_GATE="${RULE_PRIOR_STOCHASTIC_GATE:-false}"
RULE_PRIOR_MIN_APPLY_PROB="${RULE_PRIOR_MIN_APPLY_PROB:-0.40}"
RULE_PRIOR_MAX_APPLY_PROB="${RULE_PRIOR_MAX_APPLY_PROB:-0.98}"
RULE_PRIOR_CONFIDENCE_PROB_POWER="${RULE_PRIOR_CONFIDENCE_PROB_POWER:-1.0}"
RULE_PRIOR_STOCHASTIC_EVAL="${RULE_PRIOR_STOCHASTIC_EVAL:-false}"
RULE_PRIOR_STOCHASTIC_SEED="${RULE_PRIOR_STOCHASTIC_SEED:-20260528}"
RULE_PRIOR_PHASE_MATCH_MODE="${RULE_PRIOR_PHASE_MATCH_MODE:-current}"
RULE_PRIOR_ENABLE_GROUNDED_HOP_PHASE="${RULE_PRIOR_ENABLE_GROUNDED_HOP_PHASE:-false}"
TVA_TERMINAL_PRIOR_ENABLED="${TVA_TERMINAL_PRIOR_ENABLED:-false}"
TVA_TERMINAL_PRIOR_SD_CHECKPOINT="${TVA_TERMINAL_PRIOR_SD_CHECKPOINT:-}"
TVA_TERMINAL_PRIOR_CHECKPOINT="${TVA_TERMINAL_PRIOR_CHECKPOINT:-}"
TVA_TERMINAL_PRIOR_WEIGHT="${TVA_TERMINAL_PRIOR_WEIGHT:-0.02}"
TVA_TERMINAL_PRIOR_MAX_BONUS="${TVA_TERMINAL_PRIOR_MAX_BONUS:-0.05}"
TVA_TERMINAL_PRIOR_NORMALIZE="${TVA_TERMINAL_PRIOR_NORMALIZE:-true}"
RESUME_FROM="${RESUME_FROM:-}"
STEPS="${STEPS:-5000}"
EVAL_FREQ="${EVAL_FREQ:-2500}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-2500}"
CHECKPOINT_AT_SEED="${CHECKPOINT_AT_SEED:-true}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
SEED_PRETRAIN_UPDATES="${SEED_PRETRAIN_UPDATES:-}"
NUM_SAMPLES="${NUM_SAMPLES:-128}"
NUM_ELITES="${NUM_ELITES:-16}"
NUM_PI_TRAJS="${NUM_PI_TRAJS:-12}"
ITERATIONS="${ITERATIONS:-4}"

export PYTHONPATH="$ROOT/external/tdmpc2/tdmpc2:$ROOT:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

cat <<EOF
[hopper-v6-5k-rule-prior-smoke]
root=$ROOT
python=$PYTHON_BIN
gpu=$GPU_ID
seed=$SEED
exp_name=$EXP_NAME
steps=$STEPS
eval_freq=$EVAL_FREQ
checkpoint_freq=$CHECKPOINT_FREQ
checkpoint_at_seed=$CHECKPOINT_AT_SEED
eval_episodes=$EVAL_EPISODES
seed_pretrain_updates=${SEED_PRETRAIN_UPDATES:-default}
num_samples=$NUM_SAMPLES
num_elites=$NUM_ELITES
num_pi_trajs=$NUM_PI_TRAJS
iterations=$ITERATIONS
world_model_kind=$WORLD_MODEL_KIND
sd_online_supervision_enabled=$SD_ONLINE_SUPERVISION_ENABLED
rule_prior_enabled=$RULE_PRIOR_ENABLED
rule_library=$RULE_LIBRARY
rule_prior_weight=$RULE_PRIOR_WEIGHT
rule_prior_opposite_penalty=$RULE_PRIOR_OPPOSITE_PENALTY
rule_prior_max_bonus=$RULE_PRIOR_MAX_BONUS
rule_prior_mode=$RULE_PRIOR_MODE
rule_prior_preferred_cap=$RULE_PRIOR_PREFERRED_CAP
rule_prior_cap_penalty=$RULE_PRIOR_CAP_PENALTY
rule_prior_allowed_statuses=$RULE_PRIOR_ALLOWED_STATUSES
rule_prior_stochastic_gate=$RULE_PRIOR_STOCHASTIC_GATE
rule_prior_min_apply_prob=$RULE_PRIOR_MIN_APPLY_PROB
rule_prior_max_apply_prob=$RULE_PRIOR_MAX_APPLY_PROB
rule_prior_confidence_prob_power=$RULE_PRIOR_CONFIDENCE_PROB_POWER
rule_prior_stochastic_eval=$RULE_PRIOR_STOCHASTIC_EVAL
rule_prior_stochastic_seed=$RULE_PRIOR_STOCHASTIC_SEED
rule_prior_phase_match_mode=$RULE_PRIOR_PHASE_MATCH_MODE
rule_prior_enable_grounded_hop_phase=$RULE_PRIOR_ENABLE_GROUNDED_HOP_PHASE
tva_terminal_prior_enabled=$TVA_TERMINAL_PRIOR_ENABLED
tva_terminal_prior_sd_checkpoint=${TVA_TERMINAL_PRIOR_SD_CHECKPOINT:-none}
tva_terminal_prior_checkpoint=${TVA_TERMINAL_PRIOR_CHECKPOINT:-none}
tva_terminal_prior_weight=$TVA_TERMINAL_PRIOR_WEIGHT
tva_terminal_prior_max_bonus=$TVA_TERMINAL_PRIOR_MAX_BONUS
resume_from=${RESUME_FROM:-none}
sd_init_ckpt=$SD_INIT_CKPT

Intent:
- use the independent v6 training entry under experiments/sd_retrospective_rule_induction
- keep the retrospective rule prior switch default-off
- when explicitly enabled, apply only replay_validated v5 substate rules
- intervene only through low-strength MPPI candidate reweighting
EOF

if [ "$WORLD_MODEL_KIND" = "sd_hopper" ]; then
  test -f "$SD_INIT_CKPT"
fi
test -f "$RULE_LIBRARY"
if [ "$TVA_TERMINAL_PRIOR_ENABLED" = "true" ]; then
  test -f "$TVA_TERMINAL_PRIOR_SD_CHECKPOINT"
  test -f "$TVA_TERMINAL_PRIOR_CHECKPOINT"
fi

SD_INIT_ARG=()
if [ "$WORLD_MODEL_KIND" = "sd_hopper" ]; then
  SD_INIT_ARG=(sd_supervised_init_from="$SD_INIT_CKPT")
else
  SD_INIT_ARG=(sd_supervised_init_from=null)
fi

RESUME_ARG=()
if [ -n "$RESUME_FROM" ]; then
  test -f "$RESUME_FROM"
  RESUME_ARG=(resume_from="$RESUME_FROM")
else
  RESUME_ARG=(resume_from=null)
fi

exec "$PYTHON_BIN" -u "$ROOT/experiments/sd_retrospective_rule_induction/train_hopper_v6_rule_prior.py" \
  task=hopper-hop \
  world_model_kind="$WORLD_MODEL_KIND" \
  model_size=null \
  obs=state \
  steps="$STEPS" \
  seed="$SEED" \
  exp_name="$EXP_NAME" \
  enable_wandb=false \
  save_video=false \
  save_agent=true \
  save_csv=true \
  save_full_state=true \
  save_eval_traces=true \
  eval_episodes="$EVAL_EPISODES" \
  eval_freq="$EVAL_FREQ" \
  checkpoint_freq="$CHECKPOINT_FREQ" \
  checkpoint_at_seed="$CHECKPOINT_AT_SEED" \
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
  sd_online_supervision_enabled="$SD_ONLINE_SUPERVISION_ENABLED" \
  sd_online_supervision_lr=3e-4 \
  sd_online_supervision_one_step_weight=1.0 \
  sd_online_supervision_rollout_weight=1.0 \
  sd_online_supervision_omega_weight=0.25 \
  sd_online_supervision_contact_weight=0.25 \
  sd_online_supervision_phase_weight=2.0 \
  sd_online_planning_weight_enabled=false \
  num_samples="$NUM_SAMPLES" \
  num_elites="$NUM_ELITES" \
  num_pi_trajs="$NUM_PI_TRAJS" \
  horizon=3 \
  iterations="$ITERATIONS" \
  seed_pretrain_updates="${SEED_PRETRAIN_UPDATES:-null}" \
  "${SD_INIT_ARG[@]}" \
  "${RESUME_ARG[@]}" \
  +rule_prior_enabled="$RULE_PRIOR_ENABLED" \
  +rule_prior_library="$RULE_LIBRARY" \
  +rule_prior_weight="$RULE_PRIOR_WEIGHT" \
  +rule_prior_opposite_penalty="$RULE_PRIOR_OPPOSITE_PENALTY" \
  +rule_prior_max_bonus="$RULE_PRIOR_MAX_BONUS" \
  +rule_prior_mode="$RULE_PRIOR_MODE" \
  +rule_prior_preferred_cap="$RULE_PRIOR_PREFERRED_CAP" \
  +rule_prior_cap_penalty="$RULE_PRIOR_CAP_PENALTY" \
  +rule_prior_allowed_statuses="'$RULE_PRIOR_ALLOWED_STATUSES'" \
  +rule_prior_stochastic_gate="$RULE_PRIOR_STOCHASTIC_GATE" \
  +rule_prior_min_apply_prob="$RULE_PRIOR_MIN_APPLY_PROB" \
  +rule_prior_max_apply_prob="$RULE_PRIOR_MAX_APPLY_PROB" \
  +rule_prior_confidence_prob_power="$RULE_PRIOR_CONFIDENCE_PROB_POWER" \
  +rule_prior_stochastic_eval="$RULE_PRIOR_STOCHASTIC_EVAL" \
  +rule_prior_stochastic_seed="$RULE_PRIOR_STOCHASTIC_SEED" \
  +rule_prior_phase_match_mode="$RULE_PRIOR_PHASE_MATCH_MODE" \
  +rule_prior_enable_grounded_hop_phase="$RULE_PRIOR_ENABLE_GROUNDED_HOP_PHASE" \
  +tva_terminal_prior_enabled="$TVA_TERMINAL_PRIOR_ENABLED" \
  +tva_terminal_prior_sd_checkpoint="'$TVA_TERMINAL_PRIOR_SD_CHECKPOINT'" \
  +tva_terminal_prior_checkpoint="'$TVA_TERMINAL_PRIOR_CHECKPOINT'" \
  +tva_terminal_prior_weight="$TVA_TERMINAL_PRIOR_WEIGHT" \
  +tva_terminal_prior_max_bonus="$TVA_TERMINAL_PRIOR_MAX_BONUS" \
  +tva_terminal_prior_normalize="$TVA_TERMINAL_PRIOR_NORMALIZE"
