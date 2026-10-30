#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/grpo_vast_common.sh"
activate_venv
for key in ACTOR_CHECKPOINT CORPUS_CONFIG READER_URL READER_API_KEY READER_MODEL; do require_value "$key"; done

INDEX_DIR="${CHAT_INDEX_DIR:-$(python - "$CORPUS_CONFIG" <<'PY'
import json, sys
config = json.load(open(sys.argv[1], encoding='utf-8'))
entries = config.get('corpora', config)
print(next(iter(entries.values()))['index_dir'])
PY
)}"
TRIPLE_CACHE="${CHAT_TRIPLE_CACHE:-$(python - "$CORPUS_CONFIG" <<'PY'
import json, sys
config = json.load(open(sys.argv[1], encoding='utf-8'))
entries = config.get('corpora', config)
print(next(iter(entries.values())).get('triple_cache', ''))
PY
)}"

CUDA_VISIBLE_DEVICES="$TRAIN_GPU" grasp-grpo-chat \
  --checkpoint "$ACTOR_CHECKPOINT" --index-dir "$INDEX_DIR" --triple-cache "$TRIPLE_CACHE" \
  --reader-url "$READER_URL" --reader-api-key "$READER_API_KEY" --reader-model "$READER_MODEL" \
  --load-in-4bit
