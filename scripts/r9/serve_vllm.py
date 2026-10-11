#!/usr/bin/env python3
"""Launch a frozen-constant vLLM OpenAI server for R9. Never pkill python."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from expolicy.r9_lab.constants import (
    DTYPE,
    GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN,
    OUTPUT_ROOT,
    model_by_id,
    vllm_port_for_model,
)
from expolicy.r9_lab.watchdog_util import write_status_live


def build_vllm_cmd(
    *,
    model_path: str,
    served_name: str,
    port: int,
    dtype: str = DTYPE,
    max_model_len: int = MAX_MODEL_LEN,
    gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION,
    python_bin: str | None = None,
) -> list[str]:
    py = python_bin or sys.executable
    return [
        py,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--served-model-name",
        served_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        dtype,
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--generation-config",
        "vllm",
    ]


def wait_ready(port: int, *, timeout: float = 600.0) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=5.0)
            if r.status_code == 200:
                return
            last = f"status={r.status_code}"
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(2.0)
    raise RuntimeError(f"vLLM not ready on port {port} within {timeout}s ({last})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--gpu", default="4")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=GPU_MEMORY_UTILIZATION)
    p.add_argument("--wait-ready", action="store_true")
    p.add_argument("--wait-timeout", type=float, default=600.0)
    p.add_argument("--foreground", action="store_true")
    args = p.parse_args(argv)
    spec = model_by_id(args.model_id)
    if not spec.local_path.exists():
        raise SystemExit(f"local model missing: {spec.local_path}")
    port = int(args.port if args.port is not None else vllm_port_for_model(spec.model_id))
    log_dir = OUTPUT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"vllm_{spec.model_id}_{port}.log"
    cmd = build_vllm_cmd(
        model_path=str(spec.local_path),
        served_name=spec.served_name,
        port=port,
        gpu_memory_utilization=float(args.gpu_memory_utilization),
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    log_handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=str(OUTPUT_ROOT),
    )
    status = log_dir / f"vllm_{spec.model_id}" / "STATUS_LIVE.json"
    write_status_live(
        status,
        pid=proc.pid,
        gpu=str(args.gpu),
        stage="vllm",
        extra={"port": port, "served_name": spec.served_name, "log": str(log_path), "cmd": cmd},
    )
    print(json_dumps({"pid": proc.pid, "port": port, "log": str(log_path), "cmd": cmd}))
    try:
        if args.wait_ready:
            wait_ready(port, timeout=float(args.wait_timeout))
        if args.foreground:
            return int(proc.wait())
    except Exception:
        # Do not pkill; only the recorded PID may be stopped by watchdog.
        raise
    return 0


def json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, default=str)


if __name__ == "__main__":
    raise SystemExit(main())
