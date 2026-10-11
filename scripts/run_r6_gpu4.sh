#!/usr/bin/env bash
set -euo pipefail
cd /data/ppnm/expolicy
export PYTHONUNBUFFERED=1
export EXPOLICY_RETRIEVER_URL="${EXPOLICY_RETRIEVER_URL:-http://127.0.0.1:18080}"
PY="${PY:-/data/ppnm/expolicy/.venv/bin/python}"
CUDA_VISIBLE_DEVICES=4 "$PY" scripts/run_r6_ava_stage1.py \
  --condition exact_ava \
  --seed 0 \
  --split outputs/r6_ava/split_r6_ava.json \
  --budget 48 \
  --batch-size 8 \
  --num-candidates 8 \
  --resume
