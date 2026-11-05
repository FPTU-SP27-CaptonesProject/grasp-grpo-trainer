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
  LOG_SYNC_PID=""
  LOG_RUN_DIR="$dir"
  # Live dashboard: push small logs (no checkpoints) to a private HF dataset every LOG_SYNC_SECONDS.
  if [[ "${LOG_SYNC:-1}" == "1" ]]; then
    python -m grasp_grpo.log_sync --run-dir "$dir" --repo "${LOG_REPO:-doancuavahk/grasp-grpo-logs}" \
      --every "${LOG_SYNC_SECONDS:-180}" >> "$dir/logs/log_sync.log" 2>&1 &
    LOG_SYNC_PID=$!
  fi
  trap stop_logging EXIT
}

stop_logging() {
  kill "$GPU_MONITOR_PID" ${LOG_SYNC_PID:+"$LOG_SYNC_PID"} 2>/dev/null || true
  if [[ -n "$LOG_SYNC_PID" ]]; then  # last upload so the dashboard sees the final state
    python -m grasp_grpo.log_sync --run-dir "$LOG_RUN_DIR" --repo "${LOG_REPO:-doancuavahk/grasp-grpo-logs}" --once \
      >> "$LOG_RUN_DIR/logs/log_sync.log" 2>&1 || true
  fi
}

# Default bf16 base: generation is 2-3x faster than the nf4 base (bitsandbytes dequantises every matmul).
# LOAD_IN_4BIT=1 loads nf4 like the QLoRA SFT run (less memory, slower).
quant_args() {
  QUANT_ARGS=()
  [[ "${LOAD_IN_4BIT:-0}" == "1" ]] && QUANT_ARGS=(--load-in-4bit)
  return 0
}

# Lazy triple extraction (Project A extractor on Bedrock) for retrieved documents missing from the cache.
extract_args() {
  EXTRACT_ARGS=()
  if [[ "${EXTRACT_TRIPLES:-0}" == "1" ]]; then
    require_value BEDROCK_API_KEY
    export BEDROCK_API_KEY
    [[ -n "${BEDROCK_BASE_URL:-}" ]] && export BEDROCK_BASE_URL
    EXTRACT_ARGS=(--extract-triples --extract-budget-usd "${EXTRACT_BUDGET_USD:-10}")
  fi
  return 0
}

reader_args() {
  READER_ARGS=(--reader-url "$READER_URL" --reader-api-key "$READER_API_KEY" --reader-model "$READER_MODEL")
}

# Proxy generation on the vLLM policy server (scripts/run_policy_vast.sh); empty POLICY_URL = HF generate.
policy_args() {
  POLICY_ARGS=()
  [[ -n "${POLICY_URL:-}" ]] && POLICY_ARGS=(--policy-url "$POLICY_URL" --policy-api-key "$READER_API_KEY" --policy-model sft)
  return 0
}

# Flags shared by every grasp-train-grpo call, built from vast.env.
train_args() {
  reader_args
  quant_args
  extract_args
  policy_args
  TRAIN_ARGS=(--sft-checkpoint "$SFT_CHECKPOINT" --corpus-config "$CORPUS_CONFIG" "${READER_ARGS[@]}" "${QUANT_ARGS[@]}" "${EXTRACT_ARGS[@]}" "${POLICY_ARGS[@]}"
    --reader-protocol "${READER_PROTOCOL:-grasp}" --workers "$WORKERS"
    --branch-factor "$BRANCH_FACTOR" --max-steps "$MAX_STEPS" --max-leaves "$MAX_LEAVES"
    --alpha "$ALPHA" --shaping-scale "$SHAPING_SCALE" --retrieval-cost "${RETRIEVAL_COST:-0.0}"
    --rd-gain-weight "$RD_GAIN_WEIGHT" --qf-gain-weight "$QF_GAIN_WEIGHT"
    --ki-hit-weight "$KI_HIT_WEIGHT" --ki-miss-penalty "$KI_MISS_PENALTY"
    --learning-rate "$LEARNING_RATE" --warmup-steps "$WARMUP_STEPS"
    --clip-low "$CLIP_LOW" --clip-high "$CLIP_HIGH" --kl-beta "$KL_BETA")
}
