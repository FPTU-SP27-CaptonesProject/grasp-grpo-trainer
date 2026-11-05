#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/grpo_vast_common.sh"
require_value READER_API_KEY
READER_VENV="${READER_VENV:-/workspace/grasp-reader-venv}"
if [[ ! -f "$READER_VENV/bin/activate" ]]; then
  echo "Reader virtual environment not found: $READER_VENV. Run scripts/setup_vast.sh." >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$READER_VENV/bin/activate"

# FP8 = the reader that generated the SFT data and ran the 2026-11-05 SFT eval.
CUDA_VISIBLE_DEVICES="$READER_GPU" exec vllm serve "$READER_MODEL" \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 --quantization "${READER_QUANTIZATION:-fp8}" \
  --max-model-len 8192 --gpu-memory-utilization "${READER_GPU_UTIL:-0.52}" --api-key "$READER_API_KEY"
