#!/usr/bin/env bash
# Extract triples for the top PREFETCH_DEPTH documents of every training and eval question before a run,
# so rollouts do not wait on Bedrock. PREFETCH_DRY_RUN=1 only counts missing documents and estimates USD.
# Spending is capped by PREFETCH_BUDGET_USD (default 40). Rerunning resumes: cached documents are skipped.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/grpo_vast_common.sh"
activate_venv
for key in QA_FILE CORPUS_CONFIG EVAL_QA EVAL_IDS OUTPUT_ROOT; do require_value "$key"; done
LOG="$OUTPUT_ROOT/prefetch_triples.log"
mkdir -p "$OUTPUT_ROOT"
args=(--qa "$QA_FILE" --eval-qa "$EVAL_QA" --eval-ids "$EVAL_IDS" --corpus-config "$CORPUS_CONFIG"
  --depth "${PREFETCH_DEPTH:-20}" --extract-budget-usd "${PREFETCH_BUDGET_USD:-40}" --extract-concurrency "${PREFETCH_CONCURRENCY:-48}")
[[ "${PREFETCH_DRY_RUN:-0}" == "1" ]] && args+=(--dry-run)
[[ -n "${PREFETCH_LIMIT:-}" ]] && args+=(--limit "$PREFETCH_LIMIT")
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" grasp-prefetch-triples "${args[@]}" 2>&1 | tee -a "$LOG"
