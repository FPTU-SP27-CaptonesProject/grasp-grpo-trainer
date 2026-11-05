#!/usr/bin/env bash
# vLLM server for the 3B proxy: base model + the SFT LoRA (served as "sft"). The trainer loads each new
# adapter at runtime (load_lora_adapter), so rollouts always sample from the current policy.
# Start it AFTER the reader when both share POLICY_GPU=READER_GPU: each vLLM takes its share of the card
# at startup (READER_GPU_UTIL + POLICY_GPU_UTIL must stay below ~0.93).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/grpo_vast_common.sh"
for key in READER_API_KEY SFT_CHECKPOINT; do require_value "$key"; done
READER_VENV="${READER_VENV:-/workspace/grasp-reader-venv}"
if [[ ! -f "$READER_VENV/bin/activate" ]]; then
  echo "vLLM virtual environment not found: $READER_VENV. Run scripts/setup_vast.sh." >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$READER_VENV/bin/activate"

BASE_MODEL="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_model_name_or_path"])' "$SFT_CHECKPOINT/adapter_config.json")"
RANK="$(python -c 'import json,sys; c=json.load(open(sys.argv[1])); print(max([c["r"], *c.get("rank_pattern", {}).values()]))' "$SFT_CHECKPOINT/adapter_config.json")"
PORT="${POLICY_PORT:-8001}"

# Prompts are at most 3072 tokens (model._prompt_ids) plus 256 generated.
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True CUDA_VISIBLE_DEVICES="${POLICY_GPU:-0}" exec vllm serve "$BASE_MODEL" \
  --host 127.0.0.1 --port "$PORT" --dtype bfloat16 --max-model-len 3584 \
  --enable-lora --max-lora-rank "$RANK" --max-loras 2 --max-cpu-loras 4 \
  --lora-modules "sft=$SFT_CHECKPOINT" \
  --gpu-memory-utilization "${POLICY_GPU_UTIL:-0.40}" --api-key "$READER_API_KEY"
