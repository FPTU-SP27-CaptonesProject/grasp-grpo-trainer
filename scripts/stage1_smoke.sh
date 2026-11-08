#!/usr/bin/env bash
# Stage 1: 20 steps x 4 questions; checks finite loss, ratio ~1, small KL, non-zero gradients, LoRA moved.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/grpo_vast_common.sh"
activate_venv
for key in SFT_CHECKPOINT QA_FILE CORPUS_CONFIG READER_URL READER_API_KEY READER_MODEL OUTPUT_ROOT; do require_value "$key"; done
RUN="$OUTPUT_ROOT/stage1"
start_logging "$RUN"
train_args
CUDA_VISIBLE_DEVICES="$TRAIN_DEVICES" grasp-train-grpo "${TRAIN_ARGS[@]}" \
  --qa "$QA_FILE" --output-dir "$RUN" --iterations 20 --questions-per-iteration 4 --save-every 20
grasp-stage-gate --stage 1 --run-dir "$RUN" --sft-checkpoint "$SFT_CHECKPOINT"
