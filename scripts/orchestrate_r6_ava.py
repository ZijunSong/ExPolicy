#!/usr/bin/env python3
"""R6 AVA orchestrator: CPU gates then 4-GPU Stage-1 on GPU 4-7."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data/ppnm/expolicy")
PY = str(ROOT / ".venv" / "bin" / "python")
OUT = ROOT / "outputs" / "r6_ava"
LOG = OUT / "logs"
PORT = 18080


def log(msg: str) -> None:
    line = f"[orch {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "orchestrate.log").open("a", encoding="utf-8").write(line + "\n")


def run(cmd: list[str], name: str) -> None:
    log(" ".join(cmd))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["EXPOLICY_RETRIEVER_URL"] = f"http://127.0.0.1:{PORT}"
    rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
    if rc != 0:
        raise SystemExit(f"{name} failed rc={rc}")


def retriever_ok() -> bool:
    try:
        import httpx

        r = httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=5.0)
        return r.status_code == 200 and bool(r.json().get("ok"))
    except Exception:
        return False


def main() -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    log("==== P0 novelty already written; P1 audit ====")
    run([PY, str(ROOT / "scripts" / "audit_r6_pipeline.py")], "audit")
    audit = (OUT / "audit" / "R6_PIPELINE_AUDIT.json")
    import json

    if not json.loads(audit.read_text()).get("PASS"):
        raise SystemExit("P1 audit FAIL — stop before AVA")

    log("==== P2 case analysis ====")
    run([PY, str(ROOT / "scripts" / "analyze_r5_4_cases_for_r6.py")], "cases")
    gate = json.loads((OUT / "case_analysis" / "CASE_GATE.json").read_text())
    if not gate.get("ENTER_R6"):
        raise SystemExit("P2 gate: q×p Spearman high — do not enter AVA")

    log("==== P3 split + tests ====")
    run([PY, str(ROOT / "scripts" / "build_r6_ava_split.py")], "split")
    run(
        [
            PY,
            "-m",
            "pytest",
            "-q",
            "tests/test_r6_clone_isolation.py",
            "tests/test_r6_no_future_leakage.py",
            "tests/test_r6_candidate_unique.py",
            "tests/test_r6_shadow_determinism.py",
            "tests/test_r6_router_freeze.py",
        ],
        "pytest",
    )

    t0 = time.time()
    while not retriever_ok():
        if time.time() - t0 > 3600:
            raise SystemExit("retriever not healthy")
        log("waiting for retriever :18080")
        time.sleep(10)
    log("retriever healthy")

    run(
        [PY, str(ROOT / "scripts" / "run_r6_ava_stage1.py"), "--phase", "prep", "--root", str(OUT)],
        "prep",
    )

    if not (OUT / "base_eval_shadow" / "DONE").exists():
        log("==== shadow base_eval 4-way GPU 4-7 ====")
        run(
            [
                PY,
                str(ROOT / "scripts" / "run_r6_ava_stage1.py"),
                "--phase",
                "shadow_base",
                "--gpus",
                "4,5,6,7",
                "--resume",
                "--root",
                str(OUT),
            ],
            "shadow_base",
        )
    else:
        log("skip shadow_base (DONE)")

    if not (OUT / "MASK_EVAL.DONE").exists():
        log("==== mask_eval 4-way GPU 4-7 ====")
        run(
            [
                PY,
                str(ROOT / "scripts" / "run_r6_ava_stage1.py"),
                "--phase",
                "mask_eval",
                "--gpus",
                "4,5,6,7",
                "--resume",
                "--root",
                str(OUT),
            ],
            "mask_eval",
        )
    else:
        log("skip mask_eval (DONE)")

    log("==== Stage-1 GPU wrappers + watchdog ====")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["EXPOLICY_RETRIEVER_URL"] = f"http://127.0.0.1:{PORT}"
    rc = subprocess.call([PY, str(ROOT / "scripts" / "watchdog_r6_ava.py")], cwd=str(ROOT), env=env)
    if rc != 0:
        raise SystemExit(f"watchdog failed rc={rc}")
    if not (OUT / "STAGE1_JOBS.DONE").exists():
        raise SystemExit("Stage-1 wrappers did not finish cleanly")
    log("==== analyze Stage-1 ====")
    run(
        [
            PY,
            str(ROOT / "scripts" / "analyze_r6_ava_stage1.py"),
            "--root",
            str(OUT),
            "--out",
            str(OUT / "R6_STAGE1_MASTER.md"),
        ],
        "analyze",
    )
    log("R6 Stage-1 complete")


if __name__ == "__main__":
    main()
