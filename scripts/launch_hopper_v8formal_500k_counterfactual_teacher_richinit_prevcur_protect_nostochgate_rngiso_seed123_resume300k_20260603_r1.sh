#!/usr/bin/env bash
set -euo pipefail

# Continue the completed 300k no-gate rich-init run from cycle 060 to cycle 100.
export EXP_PREFIX="tdmpc2_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1"
export OUTPUT_ROOT="results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1"
export CYCLES=100
export TEACHER_RULES="results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1/cycle_060/rules/library_closed_loop_cycle_060_confidence.json"
export PROTECTED_TEACHER_ORIGINS="teacher_core_standard_hop,sd_tva_counterfactual_teacher_core_v1,sd_tva_counterfactual_teacher_rich_init_v1"

BASE_300K="tdmpc2_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1_cycle060_target300059"
BASE_ROOT="results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_300k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1/cycle_060"

exec "$(dirname "${BASH_SOURCE[0]}")/launch_hopper_v8formal_200k_counterfactual_teacher_retain8_prevcur_protect_nostochgate_rngiso_seed123_20260531_r1.sh" \
  --start-cycle 61 \
  --initial-sd-checkpoint "${BASE_ROOT}/sd_explainers/sd_explainer_cycle_060.pt" \
  --initial-tva-checkpoint "${BASE_ROOT}/tva/hopper_tva_cycle_060.pt" \
  --initial-tva-terminal-prior-policy strong \
  --initial-resume "1:logs/hopper-hop/1/${BASE_300K}/models/final.full.pt" \
  --initial-resume "2:logs/hopper-hop/2/${BASE_300K}/models/final.full.pt" \
  --initial-resume "3:logs/hopper-hop/3/${BASE_300K}/models/final.full.pt" \
  "$@"
