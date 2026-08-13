#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
CKPT="${1:-outputs/static_base/agent_v000}"
python -m expolicy.cli.main evaluate "$CKPT" --config configs/experiment/base.yaml --split validation --stage 0 --limit 50
