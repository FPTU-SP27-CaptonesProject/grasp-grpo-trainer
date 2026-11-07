#!/usr/bin/env bash
# Stage 0: SFT baseline on the fixed eval set with both reader protocols, plus speed and signal probe.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/grpo_vast_common.sh"
activate_venv
for key in SFT_CHECKPOINT CORPUS_CONFIG READER_URL READER_API_KEY READER_MODEL EVAL_QA EVAL_IDS OUTPUT_ROOT; do require_value "$key"; done
# Quick check (e.g. after a generation change): STAGE0_LIMIT=40 STAGE0_PROBE=20 evaluates only the first
# 40 eval questions and probes 20 trees, into stage0_quick/ so the full stage0/ report is kept.
LIMIT="${STAGE0_LIMIT:-}"
PROBE="${STAGE0_PROBE:-50}"
RUN="$OUTPUT_ROOT/stage0"
limit=()
if [[ -n "$LIMIT" ]]; then
  RUN="$OUTPUT_ROOT/stage0_quick"
  limit=(--limit "$LIMIT")
  (( PROBE > LIMIT )) && PROBE="$LIMIT"
fi
start_logging "$RUN"
reader_args
quant_args
extract_args
policy_args
# STAGE0_PROTOCOLS="grasp legacy" also re-measures the first GRPO run's reader protocol (F1 0.366 on 2026-11-05).
for protocol in ${STAGE0_PROTOCOLS:-grasp}; do
  probe=()
  [[ "$protocol" == grasp ]] && probe=(--signal-probe "$PROBE")
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" grasp-eval-policy \
    --checkpoint "$SFT_CHECKPOINT" --qa "$EVAL_QA" --eval-ids "$EVAL_IDS" --corpus-config "$CORPUS_CONFIG" \
    "${READER_ARGS[@]}" "${QUANT_ARGS[@]}" "${EXTRACT_ARGS[@]}" "${POLICY_ARGS[@]}" --reader-protocol "$protocol" "${limit[@]}" \
    --max-steps "$MAX_STEPS" --workers "$WORKERS" --max-leaves "$MAX_LEAVES" \
    --alpha "$ALPHA" --shaping-scale "$SHAPING_SCALE" "${probe[@]}" \
    --output "$RUN/stage0_${protocol}.json"
done
grasp-stage-gate --stage 0 --run-dir "$RUN"
[[ -n "$LIMIT" ]] && echo "Quick run on $LIMIT questions: F1 is indicative only; keep SFT_F1 from the full stage0/ report."
echo "Set SFT_F1 in config/vast.env to the f1 above in $RUN/stage0_grasp.json"
