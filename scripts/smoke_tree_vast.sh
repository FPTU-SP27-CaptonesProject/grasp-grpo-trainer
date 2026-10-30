#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/grpo_vast_common.sh"
activate_venv
for key in SFT_CHECKPOINT QA_FILE CORPUS_CONFIG READER_URL READER_API_KEY READER_MODEL; do require_value "$key"; done

mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" grasp-grpo-rollout \
  --checkpoint "$SFT_CHECKPOINT" --qa "$QA_FILE" --corpus-config "$CORPUS_CONFIG" \
  --reader-url "$READER_URL" --reader-api-key "$READER_API_KEY" --reader-model "$READER_MODEL" \
  --output "$OUTPUT_DIR/smoke-tree.json" --limit 1 \
  --branch-factor "$BRANCH_FACTOR" --max-steps "$MAX_STEPS" --max-leaves "$MAX_LEAVES" \
  --load-in-4bit
