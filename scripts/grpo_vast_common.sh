#!/usr/bin/env bash
# Shared environment loader for the Vast.ai command wrappers.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ENV_FILE="${ENV_FILE:-$PROJECT_DIR/config/vast.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE. Copy config/vast.env.example to config/vast.env and edit it." >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$ENV_FILE"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
export PYTHONUNBUFFERED=1
# Model downloads (vLLM reader, base model, E5) and temp files stay off the small container root disk.
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}" TMPDIR="${TMPDIR:-/workspace/tmp}"   VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/workspace/.cache/vllm}" XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0
mkdir -p "$HF_HOME" "$TMPDIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

require_value() {
  local key="$1"
  if [[ -z "${!key:-}" || "${!key}" == "<"* ]]; then
    echo "Set $key in $ENV_FILE" >&2
    exit 2
  fi
}

activate_venv() {
  if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "Virtual environment not found: $VENV_DIR. Run scripts/setup_vast.sh first." >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
}

# Tee all output to DIR/logs/console.log and sample nvidia-smi every 30 s until the script exits.
start_logging() {
  local dir="$1"
  mkdir -p "$dir/logs"
  exec > >(tee -a "$dir/logs/console.log") 2>&1
  local gpu_log="$dir/logs/gpu_metrics.csv"
  [[ -f "$gpu_log" ]] || echo "timestamp,index,name,gpu_util_percent,memory_used_mib,memory_total_mib,temperature_c,power_w" > "$gpu_log"
  (
    while true; do
      nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits >> "$gpu_log" || true
      sleep 30
    done
  ) &
  GPU_MONITOR_PID=$!
  trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT
}

reader_args() {
  READER_ARGS=(--reader-url "$READER_URL" --reader-api-key "$READER_API_KEY" --reader-model "$READER_MODEL")
}

# Flags shared by every grasp-train-grpo call, built from vast.env.
train_args() {
  reader_args
  TRAIN_ARGS=(--sft-checkpoint "$SFT_CHECKPOINT" --corpus-config "$CORPUS_CONFIG" "${READER_ARGS[@]}"
    --reader-protocol "${READER_PROTOCOL:-grasp}" --workers "$WORKERS"
    --branch-factor "$BRANCH_FACTOR" --max-steps "$MAX_STEPS" --max-leaves "$MAX_LEAVES"
    --alpha "$ALPHA" --shaping-scale "$SHAPING_SCALE" --retrieval-cost "${RETRIEVAL_COST:-0.0}"
    --rd-gain-weight "$RD_GAIN_WEIGHT" --qf-gain-weight "$QF_GAIN_WEIGHT"
    --ki-hit-weight "$KI_HIT_WEIGHT" --ki-miss-penalty "$KI_MISS_PENALTY"
    --learning-rate "$LEARNING_RATE" --warmup-steps "$WARMUP_STEPS"
    --clip-low "$CLIP_LOW" --clip-high "$CLIP_HIGH" --kl-beta "$KL_BETA")
}
