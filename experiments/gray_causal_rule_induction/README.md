# Gray Causal Rule Induction

This experiment directory is independent from `experiments/sd_retrospective_rule_induction`.

The first-stage code audits existing Hopper rule libraries and evidence reports. It does not modify SD checkpoints, TD-MPC2 training code, launch scripts, or source rule libraries.

Current entry points:

- `audit_gray_rule_certificates.py`: build read-only gray certificates.
- `export_gray_filtered_rule_libraries.py`: export loader-compatible filtered rule libraries under a gray output directory.
- `eval_gray_rule_variants.py`: run eval-only comparisons of original and gray-filtered libraries on saved Hopper checkpoints.

Example:

```bash
python experiments/gray_causal_rule_induction/audit_gray_rule_certificates.py \
  --rule-library results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1/cycle_075/rules/library_closed_loop_cycle_075_confidence.json \
  --rule-effect-audit results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1/cycle_075/reports/rule_effect_audit.json \
  --tva-shadow-update results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1/cycle_075/reports/tva_shadow_update.json \
  --tva-consequence-gate results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1/cycle_075/reports/tva_shadow_consequence_gated_update.json \
  --tva-terminal-gate results/sd_retrospective_rule_induction/closed_loop/hopper_v8formal_500k_counterfactual_teacher_richinit_prevcur_protect_nostochgate_rngiso_seed123_resume300k_20260603_r1/cycle_075/reports/tva_terminal_prior_gate.json \
  --output-json results/gray_causal_rule_induction/cycle075_gray_rule_certificates.json \
  --output-md results/gray_causal_rule_induction/cycle075_gray_rule_certificates.md
```

Eval-only smoke example:

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python -u \
  experiments/gray_causal_rule_induction/eval_gray_rule_variants.py \
  --eval-episodes 3 \
  --seeds 1 2 3 \
  --gpus 1 2 3 \
  --mujoco-gl glfw
```

The eval script also accepts `--rule-cycle`, `--tva-cycle`, and
`--checkpoint-cycle`. For example, after cycle081 checkpoints are available:

```bash
/home/anaconda3/envs/sdmcts_p100/bin/python -u \
  experiments/gray_causal_rule_induction/eval_gray_rule_variants.py \
  --rule-cycle 80 \
  --tva-cycle 80 \
  --checkpoint-cycle 81 \
  --modes rule_off original_active drop_reject_active strict_active \
  --eval-episodes 3 \
  --seeds 1 2 3 \
  --gpus 1 2 3 \
  --mujoco-gl glfw
```
