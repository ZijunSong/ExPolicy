#!/usr/bin/env bash
set -euo pipefail
cd /data/ppnm/expolicy
export PYTHONUNBUFFERED=1
export EXPOLICY_RETRIEVER_URL="${EXPOLICY_RETRIEVER_URL:-http://127.0.0.1:18080}"
PY="${PY:-/data/ppnm/expolicy/.venv/bin/python}"
CUDA_VISIBLE_DEVICES=7 "$PY" scripts/run_r5_4_condition.py \
  --condition rasa_qonly \
  --seeds 0,1,2 \
  --budget 48 \
  --resume
CUDA_VISIBLE_DEVICES=7 "$PY" scripts/run_r5_4_condition.py \
  --condition rasa_nodiversity \
  --seeds 0,1,2 \
  --budget 48 \
  --resume
