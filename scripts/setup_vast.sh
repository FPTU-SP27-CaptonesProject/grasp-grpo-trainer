#!/usr/bin/env bash
# Two venvs (trainer + vLLM reader) built with uv, which also fetches Python 3.12 when the image has an older one.
# Every cache lives under /workspace: the container root disk is often only ~16GB.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
READER_VENV="${READER_VENV:-/workspace/grasp-reader-venv}"
export UV_CACHE_DIR=/workspace/.cache/uv PIP_CACHE_DIR=/workspace/.cache/pip TMPDIR=/workspace/tmp
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

uv venv -q "$VENV_DIR" --python 3.12
uv pip install -q --python "$VENV_DIR/bin/python" -e "$PROJECT_DIR[rollout,quantization]"
# Keep the trainer compatible with the pinned Transformers/PEFT stack. The
# vLLM reader has its own environment below and may use a newer Hub client.
uv pip install -q --python "$VENV_DIR/bin/python" "huggingface_hub>=0.34,<1.0"

# vLLM requires a newer Transformers stack than the GRPO trainer. Keep its
# reader service isolated so installing either environment cannot downgrade the other.
uv venv -q "$READER_VENV" --python 3.12
uv pip install -q --python "$READER_VENV/bin/python" vllm
uv cache clean -q
echo "Setup complete. Run: source $VENV_DIR/bin/activate && hf auth login"
