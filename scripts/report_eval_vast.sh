#!/usr/bin/env bash
# Thesis comparison table on one GPU: baselines (reader only) then GRASP-SFT / GRASP-GRPO (reader + policy).
#   bash scripts/report_eval_vast.sh baselines   # needs only the reader (run_reader_vast.sh)
#   bash scripts/report_eval_vast.sh grasp       # needs reader + policy server (run_policy_vast.sh)
#   bash scripts/report_eval_vast.sh table
# REPORT_IDS (default config/test_ids.json, 999 held-out dev questions) and REPORT_TAG (default: its stem) pick
# the question set; REPORT_GRPO_ADAPTER is the GRPO adapter dir. A method whose report exists is skipped, so a
# rerun resumes; each finished report is uploaded to HF (LOG_REPO reports/<tag>/) at once.
# Several GRPO checkpoints: REPORT_ADAPTERS="grasp-sft=$SFT_CHECKPOINT grasp-grpo-s100=/path ..." ... grasp
# Smoke test: REPORT_LIMIT=6 REPORT_TAG=smoke bash scripts/report_eval_vast.sh baselines
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/grpo_vast_common.sh"
activate_venv
for key in EVAL_QA CORPUS_CONFIG OUTPUT_ROOT READER_URL READER_API_KEY READER_MODEL; do require_value "$key"; done
PHASE="${1:-baselines}"
IDS="${REPORT_IDS:-$PROJECT_DIR/config/test_ids.json}"
TAG="${REPORT_TAG:-$(basename "$IDS" .json)}"
OUT="$OUTPUT_ROOT/report/$TAG"
DEVICE="${REPORT_ENCODER_DEVICE:-cuda:${RETRIEVER_GPU:-0}}"
mkdir -p "$OUT"
reader_args

upload() {
  hf upload "${LOG_REPO:-doancuavahk/grasp-grpo-logs}" "$1" "reports/$TAG/$(basename "$1")" --repo-type dataset >/dev/null \
    || echo "upload failed for $1 (kept locally)" >&2
}

# A report must cover every id in $IDS: eval_adapter_vast.sh once ran eval300 from vast.env instead (2026-11-12).
check_count() {
  python -c 'import json,sys; n=json.load(open(sys.argv[1]))["questions"]; m=len(json.load(open(sys.argv[2]))["ids"]); sys.exit(0 if n==m else f"{sys.argv[1]}: {n} questions, {sys.argv[2]} has {m}")' "$1" "$IDS"
}

case "$PHASE" in
  baselines)
    for method in ${REPORT_METHODS:-llm single single-bedrock ircot}; do
      out="$OUT/$method.json"
      [[ -s "$out" ]] && { echo "skip $method (exists)"; continue; }
      [[ "$method" == single-bedrock ]] && { require_value BEDROCK_API_KEY; export BEDROCK_API_KEY; }
      echo "== $method on $TAG"
      python -m grasp_grpo.baselines --method "$method" --qa "$EVAL_QA" --eval-ids "$IDS" --corpus-config "$CORPUS_CONFIG" \
        --encoder-device "$DEVICE" "${READER_ARGS[@]}" --max-steps "${MAX_STEPS:-5}" --workers "${REPORT_WORKERS:-16}" ${REPORT_LIMIT:+--limit "$REPORT_LIMIT"} --output "$out.tmp"
      check_count "$out.tmp" && mv "$out.tmp" "$out" && upload "$out"
    done ;;
  grasp)
    # REPORT_ADAPTERS="name=dir name=dir ..." evaluates several checkpoints (one row each; list grasp-sft yourself
    # if wanted). Two sessions with disjoint lists run side by side on one policy server.
    if [[ -n "${REPORT_ADAPTERS:-}" ]]; then
      read -ra pairs <<< "$REPORT_ADAPTERS"
    else
      require_value REPORT_GRPO_ADAPTER
      pairs=("grasp-sft=$SFT_CHECKPOINT" "grasp-grpo=$REPORT_GRPO_ADAPTER")
    fi
    for pair in "${pairs[@]}"; do
      name="${pair%%=*}"; adapter="${pair#*=}"; out="$OUT/$name.json"
      [[ -s "$out" ]] && { echo "skip $name (exists)"; continue; }
      echo "== $name ($adapter) on $TAG"
      EVAL_ADAPTER="$adapter" EVAL_IDS="$IDS" EVAL_OUTPUT="$out.tmp" EVAL_ENCODER_DEVICE="$DEVICE" EVAL_WORKERS="${REPORT_WORKERS:-16}" \
        bash "$SCRIPT_DIR/eval_adapter_vast.sh"
      check_count "$out.tmp" && mv "$out.tmp" "$out" && upload "$out"
    done ;;
  table) ;;
  *) echo "usage: $0 baselines|grasp|table" >&2; exit 2 ;;
esac
shopt -s nullglob
reports=("$OUT"/*.json)
if (( ${#reports[@]} )); then
  python -m grasp_grpo.report_table "${reports[@]}" --output "$OUT/table.md"
  upload "$OUT/table.md"
fi
