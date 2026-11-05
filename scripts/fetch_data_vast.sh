#!/usr/bin/env bash
# Download everything stages 0-4 read, in the layout config/corpora.vast.json expects.
#   SFT adapter        -> $SFT_CHECKPOINT
#   corpora + indexes  -> /workspace/data/grasp-retrieval/{2wiki_ircot,hotpotqa_abstracts_2017,wiki18}
#   triple cache       -> /workspace/data/grasp-retrieval/cache/triples.sqlite (Project A cache, shared)
#   gold-evidence QA   -> /workspace/data/grasp-gold-evidence/{dev,train_mapped}.jsonl
# Safe to rerun: files already present are skipped by `hf download`.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/grpo_vast_common.sh"
activate_venv
require_value SFT_CHECKPOINT

DATA_DIR="${DATA_DIR:-/workspace/data}"
ADAPTER_REPO="${ADAPTER_REPO:-doancuavahk/grasp-qwen2.5-3b-sft}"
RETRIEVAL_REPO="${RETRIEVAL_REPO:-doancuavahk/grasp-retrieval}"
GOLD_REPO="${GOLD_REPO:-doancuavahk/grasp-gold-evidence}"

# The container root disk is small: keep every cache under /workspace. Xet's chunk cache defaults to ~10GB.
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}" TMPDIR="${TMPDIR:-/workspace/tmp}"
export HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0
mkdir -p "$HF_HOME" "$TMPDIR" "$DATA_DIR"
free_gb=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
echo "free disk under /workspace: ${free_gb}GB"
(( free_gb >= 40 )) || { echo "need >= 40GB free under /workspace for corpora + models" >&2; exit 1; }

hf download "$ADAPTER_REPO" --local-dir "$SFT_CHECKPOINT" >/dev/null
# emb/ is not needed: DenseRetriever re-scores FAISS candidates by re-encoding them.
hf download "$RETRIEVAL_REPO" --repo-type dataset --local-dir "$DATA_DIR/grasp-retrieval" \
  --exclude "*/emb/*" --exclude ".gitattributes" >/dev/null
hf download "$GOLD_REPO" dev.jsonl train_mapped.jsonl --repo-type dataset \
  --local-dir "$DATA_DIR/grasp-gold-evidence" >/dev/null

missing=0
for f in grasp-retrieval/cache/triples.sqlite grasp-gold-evidence/dev.jsonl grasp-gold-evidence/train_mapped.jsonl \
         grasp-retrieval/{2wiki_ircot,hotpotqa_abstracts_2017,wiki18}/passages.sqlite \
         grasp-retrieval/{2wiki_ircot,hotpotqa_abstracts_2017,wiki18}/e5-large-v2/index.faiss; do
  [[ -s "$DATA_DIR/$f" ]] || { echo "missing $DATA_DIR/$f" >&2; missing=1; }
done
[[ -s "$SFT_CHECKPOINT/adapter_config.json" ]] || { echo "missing $SFT_CHECKPOINT/adapter_config.json" >&2; missing=1; }
(( missing == 0 )) || exit 1
echo "data ready under $DATA_DIR"
