#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/grpo_vast_common.sh"
activate_venv
require_value SFT_CHECKPOINT
require_value CORPUS_CONFIG

FIRST_INDEX="$(python - "$CORPUS_CONFIG" <<'PY'
import json, sys
config = json.load(open(sys.argv[1], encoding='utf-8'))
entries = config.get('corpora', config)
first = next(iter(entries.values()))
print(first['index_dir'])
PY
)"
grasp-grpo-validate --checkpoint "$SFT_CHECKPOINT" --index-dir "$FIRST_INDEX"
