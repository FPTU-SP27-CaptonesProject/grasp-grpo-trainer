#!/usr/bin/env bash
# Stage 2: 40 steps on the same 8 questions; reward must rise >= 30% or the pipeline cannot learn.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/grpo_vast_common.sh"
activate_venv
for key in SFT_CHECKPOINT QA_FILE CORPUS_CONFIG READER_URL READER_API_KEY READER_MODEL OUTPUT_ROOT; do require_value "$key"; done
RUN="$OUTPUT_ROOT/stage2"
start_logging "$RUN"
head -n 8 "$QA_FILE" > "$RUN/overfit8.jsonl"
train_args
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" grasp-train-grpo "${TRAIN_ARGS[@]}" \
  --qa "$RUN/overfit8.jsonl" --output-dir "$RUN" --iterations 40 --questions-per-iteration 8 --save-every 40
grasp-stage-gate --stage 2 --run-dir "$RUN"
