#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/grpo_vast_common.sh"
activate_venv

TRAJECTORY_DIR="${TRAJECTORY_DIR:-/workspace/data/grasp-sft/trajectories}"
CACHE_DIR="${TRIPLE_CACHE_DIR:-/workspace/data/grasp-triples}"
mkdir -p "$CACHE_DIR"
grasp-build-triple-cache --trajectories "$TRAJECTORY_DIR/2wiki.jsonl" --output "$CACHE_DIR/2wiki.sqlite" --overwrite
grasp-build-triple-cache --trajectories "$TRAJECTORY_DIR/hotpotqa.jsonl" --output "$CACHE_DIR/hotpotqa.sqlite" --overwrite
grasp-build-triple-cache --trajectories "$TRAJECTORY_DIR/nq.jsonl" --output "$CACHE_DIR/wiki18.sqlite" --overwrite
