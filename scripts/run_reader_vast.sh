#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/grpo_vast_common.sh"
activate_venv
require_value READER_API_KEY

CUDA_VISIBLE_DEVICES="$READER_GPU" exec vllm serve "$READER_MODEL" \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 \
  --gpu-memory-utilization 0.90 --api-key "$READER_API_KEY"
