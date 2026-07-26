#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export PYTHONPATH="$ROOT/external/tdmpc2/tdmpc2:$ROOT:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/home/anaconda3/envs/sdmcts_p100/bin/python}"
EXP_PREFIX="tdmpc2_v8formal_200k_corrected_coldstart_stochgate_tva_seed123_20260524_r1"
OUTPUT_ROOT="results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_corrected_coldstart_stochgate_tva_seed123_20260524_r1"
COLDSTART_RULES="results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_200k_teacher_core_init_seed123_20260522_r1/cycle_040/rules/library_corrected_label_coldstart_initial_v2.json"

exec "$PYTHON_BIN" experiments/sd_retrospective_rule_induction/run_hopper_v7_multiseed_shared_rules.py \
  --seeds 1,2,3 \
  --gpu-ids 1,2,3 \
  --cycles 40 \
  --start-cycle 1 \
  --cycle-steps 5000 \
  --exp-prefix "$EXP_PREFIX" \
  --output-root "$OUTPUT_ROOT" \
  --python-bin "$PYTHON_BIN" \
  --initial-rule-library "$COLDSTART_RULES" \
  --enable-rule-prior-cycle1 \
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
  --enable-rule-prior-stochastic-gate \
  --rule-prior-min-apply-prob 0.10 \
  --rule-prior-max-apply-prob 0.95 \
  --rule-prior-confidence-prob-power 1.0 \
  --enable-tva-terminal-prior \
  --enable-tva-terminal-prior-gate \
  --enable-tva-terminal-prior-weak-gate \
  --tva-terminal-prior-weight 0.02 \
  --tva-terminal-prior-max-bonus 0.05 \
  --tva-terminal-weak-prior-weight 0.005 \
  --tva-terminal-weak-prior-max-bonus 0.01 \
  --tva-terminal-gate-horizons 1,3 \
  --tva-terminal-gate-max-episodes 20 \
  --tva-terminal-gate-max-starts-per-episode 2 \
  --tva-terminal-gate-min-samples 30 \
  --tva-terminal-gate-min-oracle-pearson 0.55 \
  --tva-terminal-gate-min-sd-pearson 0.55 \
  --tva-terminal-gate-min-sd-r2 0.20 \
  --tva-terminal-gate-max-sd-rmse-ratio 0.90 \
  --tva-terminal-weak-gate-horizon-mode any \
  --tva-terminal-weak-gate-min-samples 30 \
  --tva-terminal-weak-gate-min-oracle-pearson 0.50 \
  --tva-terminal-weak-gate-min-sd-pearson 0.50 \
  --tva-terminal-weak-gate-min-sd-r2 0.00 \
  --tva-terminal-weak-gate-max-sd-rmse-ratio 1.00 \
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
