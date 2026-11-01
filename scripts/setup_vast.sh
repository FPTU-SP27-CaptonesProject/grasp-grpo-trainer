#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
READER_VENV="${READER_VENV:-/workspace/grasp-reader-venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e "$PROJECT_DIR[rollout,quantization]"
# Keep the trainer compatible with the pinned Transformers/PEFT stack. The
# vLLM reader has its own environment below and may use a newer Hub client.
python -m pip install "huggingface_hub>=0.34,<1.0"

# vLLM requires a newer Transformers stack than the GRPO trainer. Keep its
# reader service isolated so installing either environment cannot downgrade the other.
"$PYTHON_BIN" -m venv "$READER_VENV"
# shellcheck disable=SC1090
source "$READER_VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install vllm
echo "Setup complete. Run: hf auth login"
