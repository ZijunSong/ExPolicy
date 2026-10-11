#!/usr/bin/env bash
# Serve frozen Qwen2.5-3B-Instruct via vLLM OpenAI API. GPU must be 4-7.
set -euo pipefail
ROOT=/data/ppnm/expolicy
MODEL=/data/ppnm/models/Qwen2.5-3B-Instruct
GPU="${R8_VLLM_GPU:-4}"
PORT="${R8_VLLM_PORT:-18090}"
UTIL="${R8_VLLM_UTIL:-0.18}"
LOGDIR="${ROOT}/outputs/r8_backend_validation/logs"
mkdir -p "${LOGDIR}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
PY="${ROOT}/.venv/bin/python"
if ! "${PY}" -c "import vllm" >/dev/null 2>&1; then
  echo "vLLM missing in ${PY}; trying python on PATH"
  PY=python
fi
exec "${PY}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --served-model-name Qwen2.5-3B-Instruct \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization "${UTIL}" \
  --max-num-seqs 16 \
  --no-enable-log-requests \
  > "${LOGDIR}/vllm_gpu${GPU}.out" 2>&1
