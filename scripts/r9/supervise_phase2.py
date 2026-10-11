#!/usr/bin/env python3
"""Phase-2/3 supervisor: keep Q2 up, run 05 then farm 07 on GPUs 4-7.

Kill ONLY PIDs recorded in outputs/r9_lab/jobs/PIDS.txt whose cmdline contains r9.
Never pkill python. Never touch other users' processes.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path("/data/ppnm/expolicy")
OUT = ROOT / "outputs" / "r9_lab"
LOG = OUT / "logs"
JOBS = OUT / "jobs"
PY = ROOT / ".venv" / "bin" / "python"
STALL_SEC = 20 * 60
Q2_PORTS = {"4": 18104, "5": 18105, "6": 18106, "7": 18102}
Q2_UTIL = "0.50"


def log(msg: str) -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with (LOG / "supervise_phase2.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def cmdline(pid: int) -> str:
    p = Path(f"/proc/{pid}/cmdline")
    if not p.exists():
        return ""
    return p.read_text(errors="ignore").replace("\x00", " ")


def recorded_pids() -> list[int]:
    path = JOBS / "PIDS.txt"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if parts and parts[0].isdigit():
            out.append(int(parts[0]))
    return out


def is_our_r9(pid: int) -> bool:
    if pid <= 1 or pid == os.getpid():
        return False
    if pid not in recorded_pids():
        return False
    cmd = cmdline(pid)
    markers = (
        "05_operator_sanity",
        "07_run_assim_matrix",
        "04_foundation",
        "serve_vllm.py",
        "vllm.entrypoints.openai.api_server",
        "supervise_phase2",
        "scripts/r9/",
    )
    return any(m in cmd for m in markers)


def vllm_up(port: int) -> bool:
    try:
        import httpx

        r = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


def sparql_up() -> bool:
    try:
        import httpx

        r = httpx.get("http://127.0.0.1:3001/sparql", timeout=3.0)
        return r.status_code < 500
    except Exception:
        return False


def launch(cmd: list[str], logfile: Path, extra_env: dict | None = None) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{ROOT.parent / 'third_party' / 'LifelongAgentBench'}"
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        cmd,
        stdout=logfile.open("a"),
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(ROOT),
        start_new_session=True,
    )
    log(f"launch pid={proc.pid} cmd={' '.join(cmd)}")
    JOBS.mkdir(parents=True, exist_ok=True)
    with (JOBS / "PIDS.txt").open("a") as f:
        f.write(f"{proc.pid} {' '.join(cmd)}\n")
    return proc.pid


def port_has_vllm(port: int) -> bool:
    needle = f"--port {port}"
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        cmd = cmdline(int(proc.name))
        if needle in cmd and ("vllm.entrypoints.openai.api_server" in cmd or "serve_vllm.py" in cmd):
            return True
    return False


def ensure_q2(gpu: str, port: int) -> None:
    lock = JOBS / f"vllm_Q2_port{port}.launch"
    if vllm_up(port):
        lock.unlink(missing_ok=True)
        return
    if port_has_vllm(port):
        return
    lock = JOBS / f"vllm_Q2_port{port}.launch"
    JOBS.mkdir(parents=True, exist_ok=True)
    if lock.exists() and time.time() - lock.stat().st_mtime < 1200:
        return
    lock.write_text(str(time.time()), encoding="utf-8")
    log(f"starting Q2 vLLM gpu={gpu} port={port}")
    launch(
        [
            str(PY),
            "scripts/r9/serve_vllm.py",
            "--model-id",
            "Q2",
            "--gpu",
            gpu,
            "--port",
            str(port),
            "--gpu-memory-utilization",
            Q2_UTIL,
            "--wait-ready",
            "--wait-timeout",
            "1200",
        ],
        LOG / f"serve_Q2_gpu{gpu}_{port}.log",
        extra_env={"CUDA_VISIBLE_DEVICES": gpu},
    )


def sanity_done() -> bool:
    md = OUT / "audit" / "OPERATOR_SANITY.md"
    if not md.exists():
        return False
    text = md.read_text(encoding="utf-8")
    return "OPERATOR_SANITY_PASS" in text and "OPERATOR_SANITY_FAIL" not in text.split("**status:**")[-1][:80]


def sanity_running() -> bool:
    for pid in recorded_pids():
        if "05_operator_sanity" in cmdline(pid) and Path(f"/proc/{pid}").exists():
            return True
    return False


def matrix_complete() -> bool:
    frozen_path = OUT / "FROZEN_BATCHES.json"
    if not frozen_path.exists():
        return False
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    actions = ("DISCARD", "EPISODIC_REPLAY", "SEMANTIC_MEMORY", "SKILLOPT_EDIT")
    n = 0
    done = 0
    for batch in frozen.get("batches") or []:
        for action in actions:
            n += 1
            if (OUT / "matrix" / batch["batch_id"] / action / "DONE.json").exists():
                done += 1
    return n > 0 and done == n


def worker_running(gpu: str) -> bool:
    for pid in recorded_pids():
        if not Path(f"/proc/{pid}").exists():
            continue
        cmd = cmdline(pid)
        if "07_run_assim_matrix.py" in cmd and f"--gpu {gpu}" in cmd:
            return True
    return False


def stall_reap() -> None:
    """Kill a recorded R9 worker only if ALL of its STATUS_LIVE files are stale.

    Completed cells leave behind old STATUS_LIVE.json; using a single stale file
    would SIGTERM a worker that is still progressing on a later cell.
    """
    latest: dict[int, tuple[float, Path, str]] = {}
    for path in OUT.rglob("STATUS_LIVE.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(row.get("stage") or "") == "vllm":
            continue
        pid = row.get("pid")
        if not pid:
            continue
        pid = int(pid)
        last = float(row.get("last_progress_time") or 0)
        prev = latest.get(pid)
        if prev is None or last > prev[0]:
            latest[pid] = (last, path, str(row.get("stage") or ""))
    now = time.time()
    for pid, (last, path, stage) in latest.items():
        if not last or now - last < STALL_SEC:
            continue
        cmd = cmdline(pid)
        if "vllm.entrypoints.openai.api_server" in cmd or "serve_vllm.py" in cmd:
            continue
        if not is_our_r9(pid):
            continue
        if not Path(f"/proc/{pid}").exists():
            continue
        log(f"STALL pid={pid} file={path} stage={stage} idle={int(now - last)}s")
        try:
            os.kill(pid, signal.SIGTERM)
            log(f"SIGTERM recorded r9 pid={pid}")
        except ProcessLookupError:
            pass


def loop() -> None:
    log("phase2 supervisor start")
    while True:
        try:
            stall_reap()
            if not sparql_up():
                log("SPARQL down; trying start_lab_sparql")
                launch([str(PY), "scripts/r9/start_lab_sparql.py"], LOG / "sparql_start.log")
            ensure_q2("7", Q2_PORTS["7"])
            if not sanity_done():
                for gpu, port in Q2_PORTS.items():
                    ensure_q2(gpu, port)
            if not sanity_done() and not sanity_running() and vllm_up(Q2_PORTS["7"]):
                launch(
                    [
                        str(PY),
                        "scripts/r9/05_operator_sanity.py",
                        "--resume",
                        "--gpu",
                        "7",
                        "--port",
                        str(Q2_PORTS["7"]),
                    ],
                    LOG / "05_operator_sanity.log",
                )
            if sanity_done() and not matrix_complete():
                for gpu, port in Q2_PORTS.items():
                    ensure_q2(gpu, port)
                mapping = [
                    ("4", Q2_PORTS["4"], ["--env", "db"]),
                    ("5", Q2_PORTS["5"], ["--env", "os"]),
                    ("6", Q2_PORTS["6"], ["--env", "kg", "--shard", "0", "--shards", "2"]),
                    ("7", Q2_PORTS["7"], ["--env", "kg", "--shard", "1", "--shards", "2"]),
                ]
                for gpu, port, extra in mapping:
                    if worker_running(gpu) or not vllm_up(port):
                        continue
                    launch(
                        [
                            str(PY),
                            "scripts/r9/07_run_assim_matrix.py",
                            "--resume",
                            "--gpu",
                            gpu,
                            "--port",
                            str(port),
                            *extra,
                        ],
                        LOG / f"07_matrix_gpu{gpu}.log",
                    )
            if matrix_complete() and not (OUT / "R9_GATE_H.json").exists():
                launch(
                    [str(PY), "scripts/r9/08_analyze_gate_h.py", "--resume"],
                    LOG / "08_gate_h.log",
                )
            time.sleep(30)
        except Exception as exc:  # noqa: BLE001
            log(f"supervisor error: {exc}")
            time.sleep(30)


if __name__ == "__main__":
    loop()
