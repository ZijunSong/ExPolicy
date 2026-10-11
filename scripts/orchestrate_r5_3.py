#!/usr/bin/env python3
"""Wait for wiki-18 downloads, prepare the retriever, then run the R5.3 pipeline.

GPU 4-7, resume-on-stall, no preemption of other users' processes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data/ppnm/expolicy")
PY = str(ROOT / ".venv" / "bin" / "python")
DATA = ROOT / "data" / "wiki18"
CORPUS_GZ = DATA / "corpus" / "wiki-18.jsonl.gz"
CORPUS = DATA / "corpus" / "wiki-18.jsonl"
INDEX_DIR = DATA / "e5_hnsw"
INDEX = DATA / "e5_HNSW64.index"
MODEL = Path("/data/ppnm/models/e5-base-v2")
OUT = ROOT / "outputs" / "r5_3_external_generalization"
LOG = OUT / "logs"
PORT = 18080
GPUS = "4,5,6,7"
CORPUS_GZ_SIZE = 5123307260
# HNSW split sizes from HuggingFace metadata (part_ab is smaller than 40GiB).
EXPECTED_PARTS = {
    "part_aa": 42949672960,
    "part_ab": 32706855462,
}
PART_SIZE = EXPECTED_PARTS["part_aa"]


def log(msg: str) -> None:
    print(f"[orch {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_file(path: Path, size: int, timeout: int = 8 * 3600) -> None:
    t0 = time.time()
    last = -1
    stall = 0
    while time.time() - t0 < timeout:
        sz = path.stat().st_size if path.exists() else 0
        if sz >= size - 1024:
            log(f"ready {path} ({sz})")
            return
        if sz == last:
            stall += 1
        else:
            stall = 0
            last = sz
        log(f"wait {path.name} have={sz/1e9:.2f}GB need={size/1e9:.2f}GB stall={stall}")
        time.sleep(30)
    raise SystemExit(f"timeout waiting for {path}")


def wait_hnsw() -> list[Path]:
    t0 = time.time()
    while time.time() - t0 < 8 * 3600:
        ready: list[Path] = []
        status = []
        for name, need in EXPECTED_PARTS.items():
            path = INDEX_DIR / name
            sz = path.stat().st_size if path.exists() else 0
            status.append(f"{name}={sz/1e9:.2f}/{need/1e9:.2f}GB")
            if sz >= need - 1024:
                ready.append(path)
        if len(ready) == len(EXPECTED_PARTS):
            log(f"hnsw parts ready {[p.name for p in ready]}")
            return [INDEX_DIR / name for name in EXPECTED_PARTS]
        log("wait hnsw " + " ".join(status))
        time.sleep(30)
    raise SystemExit("timeout waiting for HNSW parts")


def prepare(parts: list[Path]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    if not INDEX.exists() or INDEX.stat().st_size < PART_SIZE:
        log(f"concat {parts} -> {INDEX}")
        with INDEX.open("wb") as out:
            for part in parts:
                with part.open("rb") as handle:
                    shutil.copyfileobj(handle, out)
        log(f"index size {INDEX.stat().st_size}")
    if not CORPUS.exists() or CORPUS.stat().st_size < 1_000_000:
        if not CORPUS_GZ.exists():
            raise SystemExit("corpus gz missing")
        log(f"decompress {CORPUS_GZ}")
        subprocess.check_call(["gzip", "-dkf", str(CORPUS_GZ)])
        if not CORPUS.exists():
            raise SystemExit("decompress failed")
        log(f"corpus size {CORPUS.stat().st_size}")
    # HuggingFace dump may be a tar wrapping wiki_dump.jsonl.
    with CORPUS.open("rb") as handle:
        head = handle.read(512)
    if b"ustar" in head:
        unpacked = DATA / "corpus" / "wiki-18.unpacked.jsonl"
        if not unpacked.exists() or unpacked.stat().st_size < 1_000_000:
            log(f"extract tar corpus {CORPUS} -> {unpacked}")
            import tarfile
            with tarfile.open(CORPUS, "r") as tf:
                member = next((m for m in tf.getmembers() if m.isfile() and m.name.endswith(".jsonl")), None)
                if member is None:
                    raise SystemExit("no jsonl member in corpus tar")
                src_file = tf.extractfile(member)
                if src_file is None:
                    raise SystemExit("cannot extract corpus member")
                with unpacked.open("wb") as out:
                    last_log = 0
                    while True:
                        chunk = src_file.read(16 * 1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        if out.tell() - last_log >= 1_000_000_000:
                            last_log = out.tell()
                            log(f"unpacked {out.tell()/1e9:.2f}GB")
        log(f"using unpacked corpus {unpacked} ({unpacked.stat().st_size})")
        meta_corpus = unpacked
    else:
        meta_corpus = CORPUS
    import json
    meta = {
        "corpus": str(meta_corpus),
        "index": str(INDEX),
        "e5_model": str(MODEL),
        "use_flat": False,
    }
    (DATA / "BACKEND_PATHS.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def retriever_healthy() -> bool:
    try:
        import httpx
        r = httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=5.0)
        return r.status_code == 200 and bool(r.json().get("ok"))
    except Exception:
        return False


def launch_retriever() -> None:
    if retriever_healthy():
        log("retriever already healthy")
        return
    logdir = LOG
    logdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY,
        str(ROOT / "expolicy" / "environments" / "public_search" / "retriever_server.py"),
        "--index_path",
        str(INDEX),
        "--corpus_path",
        str(json.loads((DATA / "BACKEND_PATHS.json").read_text(encoding="utf-8"))["corpus"]),
        "--retriever_model",
        str(MODEL),
        "--retriever_name",
        "e5",
        "--topk",
        "5",
        "--port",
        str(PORT),
        "--device",
        "cpu",
    ]
    log("launch retriever: " + " ".join(cmd))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_DATASETS_CACHE"] = str(DATA / "hf_datasets_cache")
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    handle = (logdir / "retriever.log").open("a", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
    (OUT / "retriever.pid").write_text(str(proc.pid), encoding="utf-8")
    t0 = time.time()
    while time.time() - t0 < 3600:
        if retriever_healthy():
            log(f"retriever up pid={proc.pid}")
            return
        if proc.poll() is not None:
            raise SystemExit(f"retriever exited {proc.returncode}; see logs/retriever.log")
        time.sleep(5)
    raise SystemExit("retriever did not become healthy")


def run_pipeline() -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["EXPOLICY_RETRIEVER_URL"] = f"http://127.0.0.1:{PORT}"
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["HF_DATASETS_CACHE"] = str(DATA / "hf_datasets_cache")
    common = [
        PY,
        str(ROOT / "scripts" / "run_r5_3_external.py"),
        "--resume",
        "--output-dir",
        str(OUT),
        "--retriever-url",
        f"http://127.0.0.1:{PORT}",
        "--model",
        "/data/ppnm/models/Qwen2.5-3B-Instruct",
        "--datasets",
        "hotpotqa",
        "2wiki",
        "musique",
    ]
    steps = [
        (["--phase", "smoke", "--gpu", "4"], "smoke"),
        (["--phase", "splits"], "splits"),
        (["--phase", "capability"], "capability"),
        (["--phase", "base_eval", "--gpus", GPUS], "base_eval"),
        (["--phase", "calibration"], "calibration"),
        (["--phase", "router"], "router"),
        (["--phase", "skill_validation", "--gpu", "4"], "skill_validation"),
        (["--phase", "acquire", "--gpus", GPUS, "--seeds", "0", "1", "2", "--methods", "frontier", "ace_skill_prioritized", "passive", "--budget", "48"], "acquire_pilot"),
        (["--phase", "analyze", "--stage", "pilot"], "analyze_pilot"),
    ]
    skip = {
        "splits": (OUT / "splits" / "split.json").exists() and (OUT / "SPLIT_AUDIT.md").exists(),
        "capability": (OUT / "CAPABILITY_MAP_AUDIT.md").exists(),
        "calibration": (OUT / "calibration" / "DONE").exists(),
        "router": (OUT / "ROUTER_AUDIT.md").exists(),
        "skill_validation": (OUT / "skill_validation" / "DONE").exists(),
        "analyze_pilot": (OUT / "R5_3_DECISION.json").exists() and (OUT / "acquire_pilot.DONE").exists(),
    }
    for extra, name in steps:
        if skip.get(name):
            log(f"==== skip {name} (already frozen/done) ====")
            continue
        log(f"==== {name} ====")
        cmd = common + extra
        log(" ".join(cmd))
        rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
        if rc != 0:
            raise SystemExit(f"step {name} failed rc={rc}")
        if name == "acquire_pilot":
            (OUT / "acquire_pilot.DONE").write_text("ok\n", encoding="utf-8")
    # Stage-1 gate lives in analyze (exits non-zero if fail)
    log("==== expand 10-seed ====")
    rc = subprocess.call(
        common
        + [
            "--phase",
            "acquire",
            "--gpus",
            GPUS,
            "--seeds",
            *[str(i) for i in range(10)],
            "--methods",
            "frontier",
            "ace_skill_prioritized",
            "passive",
            "--budget",
            "48",
        ],
        cwd=str(ROOT),
        env=env,
    )
    if rc != 0:
        raise SystemExit(f"10-seed acquire failed rc={rc}")
    for extra, name in [
        (["--phase", "bamboogle", "--seeds", *[str(i) for i in range(10)]], "bamboogle"),
        (["--phase", "analyze", "--stage", "final"], "analyze_final"),
    ]:
        log(f"==== {name} ====")
        rc = subprocess.call(common + extra, cwd=str(ROOT), env=env)
        if rc != 0:
            raise SystemExit(f"step {name} failed rc={rc}")
    log("R5.3 pipeline complete")


def main() -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    log("waiting for downloads")
    wait_file(CORPUS_GZ, CORPUS_GZ_SIZE)
    parts = wait_hnsw()
    prepare(parts)
    launch_retriever()
    run_pipeline()


if __name__ == "__main__":
    main()
