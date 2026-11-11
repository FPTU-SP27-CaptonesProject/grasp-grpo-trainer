#!/usr/bin/env bash
# Split the GRPO advantage into answer-F1 / recall / shaping parts on PROBE_QUESTIONS training questions
# (src/grasp_grpo/reward_probe.py). Safe beside a live run: it uses the running vLLM servers and keeps E5 on
# the CPU, so the training GPU gets no extra memory. PROBE_ADAPTER = adapter dir (default: the SFT adapter).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/grpo_vast_common.sh"
activate_venv
for key in SFT_CHECKPOINT QA_FILE CORPUS_CONFIG POLICY_URL OUTPUT_ROOT; do require_value "$key"; done
reader_args
extract_args
policy_args
out="${PROBE_OUTPUT:-$OUTPUT_ROOT/reward_probe-$(date -u +%m%d%H%M).json}"
mkdir -p "$(dirname "$out")"
args=(--sft-checkpoint "$SFT_CHECKPOINT" --qa "$QA_FILE" --corpus-config "$CORPUS_CONFIG" --encoder-device cpu
  "${READER_ARGS[@]}" "${EXTRACT_ARGS[@]}" "${POLICY_ARGS[@]}"
  --question-offset "${PROBE_OFFSET:-${QUESTION_OFFSET:-0}}" --questions "${PROBE_QUESTIONS:-48}" --workers "${PROBE_WORKERS:-8}"
  --max-steps "$MAX_STEPS" --max-leaves "$MAX_LEAVES" --alpha "$ALPHA" --shaping-scale "$SHAPING_SCALE"
  --retrieval-cost "${RETRIEVAL_COST:-0.0}" --rd-gain-weight "$RD_GAIN_WEIGHT" --qf-gain-weight "$QF_GAIN_WEIGHT"
  --ki-hit-weight "$KI_HIT_WEIGHT" --ki-miss-penalty "$KI_MISS_PENALTY" --output "$out")
[[ -n "${PROBE_ADAPTER:-}" ]] && args+=(--adapter "$PROBE_ADAPTER")
CUDA_VISIBLE_DEVICES="" python -m grasp_grpo.reward_probe "${args[@]}"
echo "report: $out"
