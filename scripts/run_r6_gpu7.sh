#!/usr/bin/env bash
set -euo pipefail
cd /data/ppnm/expolicy
export PYTHONUNBUFFERED=1
export EXPOLICY_RETRIEVER_URL="${EXPOLICY_RETRIEVER_URL:-http://127.0.0.1:18080}"
PY="${PY:-/data/ppnm/expolicy/.venv/bin/python}"
CUDA_VISIBLE_DEVICES=7 "$PY" scripts/run_r6_ava_stage1.py \
  --condition passive \
  --seeds 0 1 2 \
  --split outputs/r6_ava/split_r6_ava.json \
  --budget 48 \
  --resume
CUDA_VISIBLE_DEVICES=7 "$PY" scripts/run_r6_ava_stage1.py \
  --condition ace_skill_prioritized \
  --seeds 0 1 2 \
  --split outputs/r6_ava/split_r6_ava.json \
  --budget 48 \
  --resume
CUDA_VISIBLE_DEVICES=7 "$PY" scripts/run_r6_ava_stage1.py \
  --condition legacy_frontier \
  --seeds 0 1 2 \
  --split outputs/r6_ava/split_r6_ava.json \
  --budget 48 \
  --resume
CUDA_VISIBLE_DEVICES=7 "$PY" scripts/run_r6_ava_stage1.py \
  --condition rasa_full \
  --seeds 0 1 2 \
  --split outputs/r6_ava/split_r6_ava.json \
  --budget 48 \
  --resume
