#!/usr/bin/env bash
# Stage 3: pilot with periodic eval and early stopping; passes when eval F1 >= SFT + 0.02 with no stop rule fired.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/grpo_vast_common.sh"
activate_venv
for key in SFT_CHECKPOINT QA_FILE CORPUS_CONFIG READER_URL READER_API_KEY READER_MODEL EVAL_QA EVAL_IDS OUTPUT_ROOT; do require_value "$key"; done
RUN="$OUTPUT_ROOT/stage3"
start_logging "$RUN"
train_args
sft=()
[[ -n "${SFT_F1:-}" ]] && sft=(--sft-f1 "$SFT_F1")
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" grasp-train-grpo "${TRAIN_ARGS[@]}" \
  --qa "$QA_FILE" --output-dir "$RUN" --iterations "$ITERATIONS" --questions-per-iteration "$QUESTIONS_PER_ITERATION" \
  --eval-qa "$EVAL_QA" --eval-ids "$EVAL_IDS" --eval-every "$EVAL_EVERY" "${sft[@]}" \
  --max-hours "$MAX_HOURS" --save-every "$SAVE_EVERY"
grasp-stage-gate --stage 3 --run-dir "$RUN"
