#!/usr/bin/env python3
"""Start vLLM when /dev/shm is not writable by this user.

The machine's /dev/shm is mode 755 root:root, so multiprocessing.SemLock
fails with EACCES. Qwen3-14B is text-only, and the in-process engine does
not need a cross-process shared-memory lock. This launcher replaces that
lock before vLLM imports it. It does not change /dev/shm.
"""

from __future__ import annotations

import multiprocessing
import multiprocessing.context
import os
import runpy
import sys


class _InProcessLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def acquire(self, block=True, timeout=None):
        return True

    def release(self):
        return None


def _lock(*args, **kwargs):
    return _InProcessLock()


multiprocessing.Lock = _lock
multiprocessing.context.BaseContext.Lock = lambda self, *args, **kwargs: _InProcessLock()

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

if __name__ == "__main__":
    sys.argv = [
        "vllm.entrypoints.openai.api_server",
        "--model",
        "/data/ppnm/models/Qwen3-14B",
        "--served-model-name",
        "qwen3-14b-r9",
        "--host",
        "127.0.0.1",
        "--port",
        os.environ.get("A_V1_VLLM_PORT", "18102"),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        os.environ.get("A_V1_GPU_MEM", "0.55"),
        "--generation-config",
        "vllm",
    ]
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
