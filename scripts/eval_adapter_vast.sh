#!/usr/bin/env bash
# Greedy eval of one saved adapter on the fixed eval set, beside a live run: the adapter is loaded on the
# running policy server under its own name (unloaded on exit) and E5 stays on the CPU, so the training GPU
# gets no extra memory. Usage: EVAL_ADAPTER=<adapter dir> [EVAL_OUTPUT=<json>] bash scripts/eval_adapter_vast.sh
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/grpo_vast_common.sh"
activate_venv
for key in EVAL_ADAPTER EVAL_QA EVAL_IDS CORPUS_CONFIG POLICY_URL OUTPUT_ROOT; do require_value "$key"; done
reader_args
extract_args
adapter="$(cd -- "$EVAL_ADAPTER" && pwd)"
name="eval-$(date -u +%m%d%H%M%S)"
out="${EVAL_OUTPUT:-$OUTPUT_ROOT/eval-$(basename "$adapter")-$(date -u +%m%d%H%M).json}"
admin() { curl -sf -X POST "${POLICY_URL%/}/$1" -H "Authorization: Bearer $READER_API_KEY" -H "Content-Type: application/json" -d "$2"; }
admin load_lora_adapter "{\"lora_name\": \"$name\", \"lora_path\": \"$adapter\"}"
trap 'admin unload_lora_adapter "{\"lora_name\": \"$name\"}" || true' EXIT
CUDA_VISIBLE_DEVICES="" python -m grasp_grpo.policy_eval --checkpoint "$adapter" --qa "$EVAL_QA" --eval-ids "$EVAL_IDS" \
  --corpus-config "$CORPUS_CONFIG" --encoder-device cpu "${READER_ARGS[@]}" "${EXTRACT_ARGS[@]}" \
  --policy-url "$POLICY_URL" --policy-api-key "$READER_API_KEY" --policy-model "$name" \
  --max-steps "$MAX_STEPS" --workers "${EVAL_WORKERS:-16}" --output "$out"
echo "report: $out"
