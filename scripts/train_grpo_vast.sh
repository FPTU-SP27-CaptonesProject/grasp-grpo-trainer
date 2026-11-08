#!/usr/bin/env bash
# Stage 4: full run, normally continuing from the pilot's best adapter (ACTOR_CHECKPOINT).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/grpo_vast_common.sh"
activate_venv
for key in SFT_CHECKPOINT QA_FILE CORPUS_CONFIG OUTPUT_DIR READER_URL READER_API_KEY READER_MODEL EVAL_QA EVAL_IDS; do require_value "$key"; done
start_logging "$OUTPUT_DIR"
train_args
extra=()
[[ -n "${ACTOR_CHECKPOINT:-}" ]] && extra+=(--actor-checkpoint "$ACTOR_CHECKPOINT")
[[ -n "${SFT_F1:-}" ]] && extra+=(--sft-f1 "$SFT_F1")
CUDA_VISIBLE_DEVICES="$TRAIN_DEVICES" grasp-train-grpo "${TRAIN_ARGS[@]}" "${extra[@]}" \
  --qa "$QA_FILE" --output-dir "$OUTPUT_DIR" --iterations "$ITERATIONS" --questions-per-iteration "$QUESTIONS_PER_ITERATION" \
  --eval-qa "$EVAL_QA" --eval-ids "$EVAL_IDS" --eval-every "$EVAL_EVERY" \
  --max-hours "$MAX_HOURS" --save-every "$SAVE_EVERY"
