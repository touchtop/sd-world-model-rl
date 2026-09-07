#!/usr/bin/env bash
set -euo pipefail

OFFSET="${1:?usage: bash scripts/run_boxoban_best_shard50.sh LEVEL_OFFSET}"
if (( OFFSET < 0 || OFFSET > 950 || OFFSET % 50 != 0 )); then
  echo "LEVEL_OFFSET must be one of 0, 50, ..., 950" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

SEED=$(( OFFSET * 1009 ))
END=$(( OFFSET + 49 ))
OUT_DIR="results/sokoban_sd_world_model/repro_shards"
mkdir -p "$OUT_DIR"
OUT_FILE=$(printf "%s/gft_sdmcts_boxoban_unfiltered_test1000_shard%03d_%03d_repro_seed%d.json" "$OUT_DIR" "$OFFSET" "$END" "$SEED")

python -u experiments/sokoban_sd_world_model/evaluate_sokoban_gft_sdmcts.py \
  --levels datasets/sokoban_boxoban/boxoban-levels/unfiltered/test/000.txt \
  --level-offset "$OFFSET" \
  --max-levels 50 \
  --horizon 20 \
  --budget 4096 \
  --max-steps 80 \
  --seed "$SEED" \
  --uct-c 1.25 \
  --leaf-value tva \
  --tva-checkpoint results/sokoban_sd_world_model/sokoban_gft_online_tva_guarded_v1_file001_o0_n40_h8_b192_s40_ep300_failfinal_anchor0p85_rp1p0_seed0.json \
  --constraint-mode tier_a \
  --select-by visits \
  --revisit-penalty 1.0 \
  --transposition \
  --gft-prior gft_delta \
  --gft-prior-temperature 2.0 \
  --gft-prior-uniform-mix 0.1 \
  --gft-progress-weight 1.0 \
  --gft-goal-weight 1.0 \
  --gft-ordering-weight 0.05 \
  --gft-commitment-weight 0.05 \
  --gft-revisit-weight 1.0 \
  --gft-transport-turn-weight 0.1 \
  --gft-transport-access-weight 0.25 \
  --gft-transport-zone-weight 0.25 \
  --gft-sd-leaf-mode discounted_progress \
  --gft-sd-leaf-capacity-weight 0.05 \
  --gft-sd-leaf-ordering-weight 0.05 \
  --gft-sd-leaf-commitment-weight 0.05 \
  --tva-value-blend 0.25 \
  --tva-dynamic-high-risk-threshold 0.95 \
  --tva-dynamic-root-select \
  --tva-state-gated-blend \
  --tva-state-gated-default-blend 0.0 \
  --tva-state-gated-positive-blend 0.25 \
  --tva-state-gated-positive-risk-max 0.3 \
  --tva-state-gated-positive-delta-min 0.25 \
  --tva-state-gated-negative-blend 0.0 \
  --tva-state-gated-negative-risk-min 0.95 \
  --tva-state-gated-negative-delta-min 1.0 \
  --tva-transport-phase-blend 0.6 \
  --tva-transport-phase-high-progress-min 20.0 \
  --tva-transport-phase-low-progress-max 10.0 \
  --tva-transport-phase-commitment-min 0.25 \
  --tva-transport-phase-max-closed-loop-step 5 \
  --tva-transport-phase-require-root-eligible \
  --sd-validation-path results/sokoban_sd_world_model/grid_stock_flow_design_smoke_quick.json \
  --tva-audit-path results/sokoban_sd_world_model/gft_online_tva_guarded_v1_file001_o0_n40_ep300_failfinal_anchor0p85_rp1p0_blend0p25_trunc_audit_file002_o0_n10_h8_b192_teacher4k_seed0.json \
  --require-independent-gates \
  --out "$OUT_FILE"

