#!/usr/bin/env python3
"""R9 runtime supervisor: downloads -> vLLM -> model-screen. Kill only recorded PIDs."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/data/ppnm/expolicy")
OUT = ROOT / "outputs" / "r9_lab"
LOG = OUT / "logs"
JOBS = OUT / "jobs"
PY = ROOT / ".venv" / "bin" / "python"
MS = Path("/data/ppnm/miniconda3/envs/expolicy-r9/bin/modelscope")
STALL_SEC = 20 * 60

MODELS = {
    "Q0": Path("/data/ppnm/models/Qwen3-30B-A3B-Instruct-2507"),
    "Q1": Path("/data/ppnm/models/QwQ-32B"),
    "Q2": Path("/data/ppnm/models/Qwen3-14B"),
    "Q3": Path("/data/ppnm/models/Qwen3-32B"),
}
PORTS = {"Q0": 18100, "Q1": 18101, "Q2": 18102, "Q3": 18103}
# Prefer 4-6; 7 is often busy. Q3 already on 4.
GPUS = {"Q3": "4", "Q2": "5", "Q0": "6", "Q1": "5"}  # Q1 waits until Q2 screen done if sharing 5
UTIL = {"Q3": "0.65", "Q2": "0.50", "Q0": "0.62", "Q1": "0.62"}


def log(msg: str) -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with (LOG / "supervise.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def model_ready(mid: str) -> bool:
    p = MODELS[mid]
    if not p.exists():
        return False
    if mid == "Q3":
        return (p / "model.safetensors.index.json").exists()
    incompletes = list(p.glob("*.incomplete"))
    shards = list(p.glob("model-*.safetensors"))
    index = p / "model.safetensors.index.json"
    return index.exists() and bool(shards) and not incompletes


def ontology_ok() -> bool:
    p = Path("/data/ppnm/third_party/LifelongAgentBench/data/v0121/knowledge_graph/ontology")
    return (p / "vocab.json").exists() and (p / "fb_roles").exists()


def sparql_up() -> bool:
    try:
        import httpx

        r = httpx.get("http://127.0.0.1:3001/sparql", timeout=3.0)
        return r.status_code < 500
    except Exception:
        return False


def vllm_up(port: int) -> bool:
    try:
        import httpx

        r = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


def screen_done(mid: str, env: str) -> bool:
    return (OUT / "model_screen" / mid / env / "DONE.json").exists()


def status_live() -> dict:
    p = OUT / "STATUS_LIVE.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


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


def vllm_launch_lock(mid: str) -> Path:
    JOBS.mkdir(parents=True, exist_ok=True)
    return JOBS / f"vllm_{mid}.launch"


def maybe_start_vllm(mid: str) -> None:
    if screen_done(mid, "os") and screen_done(mid, "db"):
        kg_needed = (not screen_done(mid, "kg")) and sparql_up() and ontology_ok()
        if not kg_needed:
            return
    if vllm_up(PORTS[mid]):
        vllm_launch_lock(mid).unlink(missing_ok=True)
        return
    if not model_ready(mid):
        return
    gpu = GPUS[mid]
    # GPU5 is shared: never co-locate Q1 and Q2.
    if mid == "Q1" and vllm_up(PORTS["Q2"]):
        return
    if mid == "Q2" and vllm_up(PORTS["Q1"]):
        return
    lock = vllm_launch_lock(mid)
    if lock.exists() and time.time() - lock.stat().st_mtime < 1200:
        return
    lock.write_text(str(time.time()), encoding="utf-8")
    log(f"starting vLLM {mid} gpu={gpu}")
    launch(
        [
            str(PY),
            "scripts/r9/serve_vllm.py",
            "--model-id",
            mid,
            "--gpu",
            gpu,
            "--port",
            str(PORTS[mid]),
            "--gpu-memory-utilization",
            UTIL[mid],
            "--wait-ready",
            "--wait-timeout",
            "1200",
        ],
        LOG / f"serve_{mid}.log",
        extra_env={"CUDA_VISIBLE_DEVICES": gpu},
    )


def maybe_start_screen(mid: str, env: str) -> None:
    if screen_done(mid, env):
        return
    if not vllm_up(PORTS[mid]):
        return
    # one screen process at a time per model (shared vLLM)
    live = status_live()
    if live.get("stage") == "model_screen" and str(live.get("batch_id", "")).startswith(f"{mid}:"):
        if time.time() - float(live.get("last_progress_time") or 0) < 30:
            return
    lock = OUT / "model_screen" / mid / env / ".lock"
    if lock.exists():
        return
    log(f"starting screen {mid} {env}")
    launch(
        [
            str(PY),
            "scripts/r9/02_model_screen.py",
            "--model-id",
            mid,
            "--env",
            env,
            "--gpu",
            GPUS[mid],
            "--port",
            str(PORTS[mid]),
            "--resume",
        ],
        LOG / f"screen_{mid}_{env}.log",
    )


def _pid_is_our_r9(pid: int) -> bool:
    """Refuse to signal anything that is not an R9 job recorded this round."""
    if pid == os.getpid():
        return False
    pids_txt = JOBS / "PIDS.txt"
    recorded = pids_txt.read_text(encoding="utf-8") if pids_txt.exists() else ""
    if str(pid) not in recorded:
        return False
    cmd_path = Path(f"/proc/{pid}/cmdline")
    if not cmd_path.exists():
        return False
    cmd = cmd_path.read_text(errors="ignore").replace("\x00", " ")
    markers = ("02_model_screen.py", "serve_vllm.py", "supervise_r9", "04_foundation", "r9_lab")
    return any(m in cmd for m in markers)


def reap() -> None:
    try:
        while True:
            pid, _ = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                break
            log(f"reaped pid={pid}")
    except ChildProcessError:
        pass


def _cmdline(pid: int) -> str:
    p = Path(f"/proc/{pid}/cmdline")
    if not p.exists():
        return ""
    return p.read_text(errors="ignore").replace("\x00", " ")


def stop_vllm(mid: str) -> None:
    """Stop our vLLM for mid after its screen finished. Never touch other users."""
    model_path = str(MODELS[mid])
    port = str(PORTS[mid])
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        cmd = _cmdline(pid)
        if model_path not in cmd:
            continue
        if f"--port {port}" not in cmd:
            continue
        if "vllm.entrypoints.openai.api_server" not in cmd and "serve_vllm.py" not in cmd:
            continue
        try:
            os.kill(pid, 15)
            log(f"stop vLLM {mid} pid={pid}")
        except ProcessLookupError:
            pass
    vllm_launch_lock(mid).unlink(missing_ok=True)


def maybe_free_gpu_for_q1() -> None:
    """Q2 and Q1 share GPU5. After Q2 OS+DB screen, release Q2 for Q1."""
    if screen_done("Q1", "os") and screen_done("Q1", "db"):
        return
    if not model_ready("Q1"):
        return
    if not (screen_done("Q2", "os") and screen_done("Q2", "db")):
        return
    if vllm_up(PORTS["Q2"]) or any(
        str(MODELS["Q2"]) in _cmdline(int(p.name))
        for p in Path("/proc").iterdir()
        if p.name.isdigit()
    ):
        log("Q2 os+db screen done; stopping Q2 vLLM to free GPU5 for Q1")
        stop_vllm("Q2")


def stall_check() -> None:
    live = status_live()
    if not live:
        return
    batch = str(live.get("batch_id") or "")
    if ":" in batch:
        mid, env = batch.split(":", 1)
        if screen_done(mid, env):
            return
    last = float(live.get("last_progress_time") or 0)
    if last and time.time() - last > STALL_SEC:
        pid = live.get("pid")
        log(f"STALL detected pid={pid} stage={live.get('stage')} last={last}")
        if pid and _pid_is_our_r9(int(pid)):
            try:
                os.kill(int(pid), 15)
                log(f"sent SIGTERM to recorded pid {pid}")
            except ProcessLookupError:
                pass
        else:
            log(f"refuse stall-kill pid={pid} (not a live recorded R9 job)")


def loop() -> None:
    log("supervisor start")
    while True:
        try:
            stall_check()
            reap()
            maybe_free_gpu_for_q1()
            for mid in ("Q2", "Q0", "Q1", "Q3"):
                maybe_start_vllm(mid)
            # Prefer OS then DB; KG waits for SPARQL + ontology.
            kg_ok = sparql_up() and ontology_ok()
            for mid in ("Q3", "Q2", "Q0", "Q1"):
                if not vllm_up(PORTS[mid]):
                    continue
                envs = ("os", "db", "kg") if kg_ok else ("os", "db")
                for env in envs:
                    if not screen_done(mid, env):
                        maybe_start_screen(mid, env)
                        break
            ready = [m for m in MODELS if model_ready(m)]
            if ready and all(screen_done(m, e) for m in ready for e in ("os", "db")):
                if kg_ok:
                    if all(screen_done(m, "kg") for m in ready):
                        log("os+db+kg screens complete for downloaded models")
                else:
                    log("os+db screens complete; waiting for SPARQL/ontology for KG")
                if all(model_ready(m) for m in MODELS) and kg_ok:
                    if all(screen_done(m, e) for m in MODELS for e in ("os", "db", "kg")):
                        log("all four models fully screened")
            time.sleep(30)
        except Exception as exc:  # noqa: BLE001
            log(f"supervisor error: {exc}")
            time.sleep(30)


if __name__ == "__main__":
    loop()
