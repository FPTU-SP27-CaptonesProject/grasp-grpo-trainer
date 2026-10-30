#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/grpo_vast_common.sh"
activate_venv
for key in SFT_CHECKPOINT QA_FILE CORPUS_CONFIG OUTPUT_DIR READER_URL READER_API_KEY READER_MODEL; do require_value "$key"; done

mkdir -p "$OUTPUT_DIR/logs"
exec > >(tee -a "$OUTPUT_DIR/logs/console.log") 2>&1
GPU_LOG="$OUTPUT_DIR/logs/gpu_metrics.csv"
echo "timestamp,index,name,gpu_util_percent,memory_used_mib,memory_total_mib,temperature_c,power_w" > "$GPU_LOG"
(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits >> "$GPU_LOG" || true
    sleep 30
  done
) &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

actor_args=()
if [[ -n "${ACTOR_CHECKPOINT:-}" ]]; then actor_args=(--actor-checkpoint "$ACTOR_CHECKPOINT"); fi

CUDA_VISIBLE_DEVICES="$TRAIN_GPU" grasp-train-grpo \
  --sft-checkpoint "$SFT_CHECKPOINT" "${actor_args[@]}" \
  --qa "$QA_FILE" --corpus-config "$CORPUS_CONFIG" \
  --reader-url "$READER_URL" --reader-api-key "$READER_API_KEY" --reader-model "$READER_MODEL" \
  --output-dir "$OUTPUT_DIR" --iterations "$ITERATIONS" --questions-per-iteration "$QUESTIONS_PER_ITERATION" \
  --branch-factor "$BRANCH_FACTOR" --max-steps "$MAX_STEPS" --max-leaves "$MAX_LEAVES" \
  --learning-rate "$LEARNING_RATE" --clip-epsilon "$CLIP_EPSILON" --kl-beta "$KL_BETA" \
  --save-every "$SAVE_EVERY" --load-in-4bit
