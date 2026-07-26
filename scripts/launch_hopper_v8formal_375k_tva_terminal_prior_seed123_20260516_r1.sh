#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export PYTHONPATH="$ROOT/external/tdmpc2/tdmpc2:$ROOT:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/home/anaconda3/envs/sdmcts_p100/bin/python}"
EXP_PREFIX="tdmpc2_v8formal_375k_tva_terminal_prior_seed123_20260516_r1"
OUTPUT_ROOT="results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_375k_tva_terminal_prior_seed123_20260516_r1"
BASE_ROOT="results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_350k_tva_consequence_shadowonly_seed123_20260516_r1"

exec "$PYTHON_BIN" experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py \
  --seeds 1,2,3 \
  --gpu-ids 1,2,3 \
  --cycles 75 \
  --start-cycle 71 \
  --cycle-steps 5000 \
  --exp-prefix "$EXP_PREFIX" \
  --output-root "$OUTPUT_ROOT" \
  --python-bin "$PYTHON_BIN" \
  --initial-rule-library "$BASE_ROOT/cycle_070/rules/library_closed_loop_cycle_070_confidence.json" \
  --initial-sd-checkpoint "$BASE_ROOT/cycle_070/sd_explainers/sd_explainer_cycle_070.pt" \
  --initial-tva-checkpoint "$BASE_ROOT/cycle_070/tva/hopper_tva_cycle_070.pt" \
  --initial-resume "1:logs/hopper-hop/1/tdmpc2_v8formal_350k_tva_consequence_shadowonly_seed123_20260516_r1_cycle070_target350069/models/final.full.pt" \
  --initial-resume "2:logs/hopper-hop/2/tdmpc2_v8formal_350k_tva_consequence_shadowonly_seed123_20260516_r1_cycle070_target350069/models/final.full.pt" \
  --initial-resume "3:logs/hopper-hop/3/tdmpc2_v8formal_350k_tva_consequence_shadowonly_seed123_20260516_r1_cycle070_target350069/models/final.full.pt" \
  --formal-eval-freq 25000 \
  --formal-checkpoint-freq 25000 \
  --formal-eval-episodes 10 \
  --rule-prior-mode penalty_only \
  --rule-prior-weight 0.05 \
  --rule-prior-opposite-penalty 0.01 \
  --rule-prior-preferred-cap 0.70 \
  --rule-prior-cap-penalty 0.01 \
  --rule-prior-allowed-statuses confidence_active \
  --rule-required-status v8_replay_safe_candidate \
  --enable-tva-terminal-prior \
  --tva-terminal-prior-weight 0.02 \
  --tva-terminal-prior-max-bonus 0.05 \
  --trajectory-score-mode stage \
  --max-online-rules-per-cycle 2 \
  --max-cycle-candidates 12 \
  --rolling-min-cycles 2 \
  --rolling-min-support 6 \
  --rolling-min-advantage 0.05 \
  --rolling-max-good-share 0.25 \
  --rolling-weaken-max-good-share 0.40 \
  --sd-epochs 3 \
  --sd-max-transitions 50000 \
  --sd-batch-size 512 \
  --sd-lr 5e-4 \
  --sd-device cpu \
  --sd-transition-residual-scale 1.0 \
  --max-states-per-episode 10 \
  --counterfactual-horizon 5 \
  --good-quantile 0.80 \
  --bad-quantile 0.20 \
  --random-per-cycle 10 \
  --selection-seed 20260507 \
  --enable-v8-transition-rules \
  --v8-transition-min-support 5 \
  --v8-transition-min-seeds 2 \
  --v8-max-false-intervention-rate 0.10 \
  --v8-max-online-transition-rules-per-cycle 2 \
  --v8-max-cycle-transition-candidates 12 \
  --enable-rule-effect-confidence \
  --confidence-default 0.56 \
  --confidence-active-threshold 0.55 \
  --confidence-harmful-threshold 0.45 \
  --confidence-positive-context-threshold 0.0 \
  --confidence-hard-harmful-threshold 0.50 \
  --confidence-hard-harmful-min-hits 100 \
  --confidence-conflict-keep-confidence-margin 0.15 \
  --confidence-conflict-keep-harmful-margin 0.10 \
  --enable-tva-audit \
  --tva-epochs 10 \
  --tva-max-transitions 5000 \
  --tva-batch-size 256 \
  --tva-lr 1e-3 \
  --tva-gamma 0.99 \
  --tva-true-coef 1.0 \
  --tva-shaped-coef 1.0 \
  --tva-device cpu \
  --tva-audit-max-records 1200 \
  --tva-audit-sample-mode label_event \
  --enable-tva-shadow-update \
  --tva-shadow-min-effect-hit-ratio 0.003 \
  --tva-shadow-max-best-family-entropy 0.65 \
  --enable-tva-consequence-gate \
  --tva-consequence-horizon 5 \
  --tva-consequence-min-relation-count 10 \
  --tva-consequence-min-outcome-margin 0.0
