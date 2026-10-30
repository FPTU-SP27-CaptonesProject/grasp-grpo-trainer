#!/usr/bin/env bash
# Shared environment loader for the Vast.ai command wrappers.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ENV_FILE="${ENV_FILE:-$PROJECT_DIR/config/vast.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE. Copy config/vast.env.example to config/vast.env and edit it." >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$ENV_FILE"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

require_value() {
  local key="$1"
  if [[ -z "${!key:-}" || "${!key}" == "<"* ]]; then
    echo "Set $key in $ENV_FILE" >&2
    exit 2
  fi
}

activate_venv() {
  if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "Virtual environment not found: $VENV_DIR. Run scripts/setup_vast.sh first." >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
}
