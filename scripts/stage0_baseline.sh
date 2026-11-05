#!/usr/bin/env bash
# Stage 0: SFT baseline on the fixed eval set with both reader protocols, plus speed and signal probe.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/grpo_vast_common.sh"
activate_venv
for key in SFT_CHECKPOINT CORPUS_CONFIG READER_URL READER_API_KEY READER_MODEL EVAL_QA EVAL_IDS OUTPUT_ROOT; do require_value "$key"; done
RUN="$OUTPUT_ROOT/stage0"
start_logging "$RUN"
reader_args
quant_args
extract_args
policy_args
# STAGE0_PROTOCOLS="grasp legacy" also re-measures the first GRPO run's reader protocol (F1 0.366 on 2026-11-05).
for protocol in ${STAGE0_PROTOCOLS:-grasp}; do
  probe=()
  [[ "$protocol" == grasp ]] && probe=(--signal-probe 50)
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" grasp-eval-policy \
    --checkpoint "$SFT_CHECKPOINT" --qa "$EVAL_QA" --eval-ids "$EVAL_IDS" --corpus-config "$CORPUS_CONFIG" \
    "${READER_ARGS[@]}" "${QUANT_ARGS[@]}" "${EXTRACT_ARGS[@]}" "${POLICY_ARGS[@]}" --reader-protocol "$protocol" \
    --max-steps "$MAX_STEPS" --workers "$WORKERS" --max-leaves "$MAX_LEAVES" \
    --alpha "$ALPHA" --shaping-scale "$SHAPING_SCALE" "${probe[@]}" \
    --output "$RUN/stage0_${protocol}.json"
done
grasp-stage-gate --stage 0 --run-dir "$RUN"
echo "Set SFT_F1 in config/vast.env to the f1 above in $RUN/stage0_grasp.json"
