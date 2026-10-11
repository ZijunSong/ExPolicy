#!/usr/bin/env python3
"""Formal foundation on selected model, calibration n=80/env. Writes R9_FOUNDATION.csv."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from expolicy.r9_lab.benchmark.official_env import OfficialLabEnv, OfficialLabInfraError, summarize_episodes
from expolicy.r9_lab.benchmark.split import load_split, records_in_bucket
from expolicy.r9_lab.constants import (
    ENVS,
    FOUNDATION_MAX,
    FOUNDATION_MIN,
    FOUNDATION_N,
    OUTPUT_ROOT,
    VLLM_PORT,
    model_by_id,
)
from expolicy.r9_lab.llm_client import VLLMClient, probe_vllm
from expolicy.r9_lab.manifest import load_manifest
from expolicy.r9_lab.resume import run_cell_once
from expolicy.r9_lab.watchdog_util import write_status_live

CSV_FIELDS = [
    "env",
    "n",
    "success",
    "n_success",
    "n_failure",
    "parser_valid_rate",
    "infra_failure_rate",
    "mean_steps",
    "tokens",
    "wall_time",
]


def classify_gate_f(rows: list[dict]) -> str:
    if any(float(r.get("infra_failure_rate") or 0) > 0.05 for r in rows):
        return "CASE_F_INFRA_FAIL"
    ok = True
    for r in rows:
        s = float(r["success"])
        if not (FOUNDATION_MIN <= s <= FOUNDATION_MAX):
            ok = False
        if int(float(r.get("n_success") or 0)) < 8 or int(float(r.get("n_failure") or 0)) < 8:
            ok = False
    return "CASE_F_PASS" if ok and len(rows) == 3 else "CASE_F_MODEL_MISMATCH"


def write_foundation_csv(out: Path, envs: list[str]) -> list[dict]:
    rows = []
    for env in envs:
        done = out / "foundation" / env / "DONE.json"
        if not done.exists():
            continue
        payload = json.loads(done.read_text(encoding="utf-8"))
        body = payload.get("payload") or payload
        rows.append({k: body.get(k, "") for k in CSV_FIELDS})
    csv_path = out / "R9_FOUNDATION.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--gpu", default=None)
    p.add_argument("--port", type=int, default=VLLM_PORT)
    p.add_argument("--env", action="append", dest="envs")
    p.add_argument("--allow-synthetic", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    out = args.output_root
    envs = args.envs or list(ENVS)
    man = load_manifest(out / "model_manifest.json")
    if not man.get("selected_model_id"):
        print(json.dumps({"status": "MODEL_LADDER_NO_FULL_PASS", "skipped": True}))
        return 0
    spec = model_by_id(str(man["selected_model_id"]))

    split = None
    if not args.dry_run:
        split = load_split(out / "split.json")
        if split.get("synthetic") and not args.allow_synthetic:
            raise RuntimeError("Refusing synthetic split.json unless --allow-synthetic")

    infra_errors: list[str] = []
    for env in envs:
        cell = out / "foundation" / env

        def _job(env=env) -> dict:
            if args.dry_run:
                return {
                    "env": env,
                    "n": FOUNDATION_N,
                    "success": "nan",
                    "n_success": 0,
                    "n_failure": 0,
                    "parser_valid_rate": "nan",
                    "infra_failure_rate": "nan",
                    "mean_steps": "nan",
                    "tokens": 0,
                    "wall_time": 0,
                    "dry_run": True,
                }
            try:
                probe_vllm(int(args.port))
            except Exception as exc:  # noqa: BLE001
                return {
                    "env": env,
                    "n": 0,
                    "success": "",
                    "n_success": 0,
                    "n_failure": 0,
                    "infra_failure_rate": 1.0,
                    "infra_error": str(exc),
                    "faked_success": False,
                }
            client = VLLMClient.from_model_id(spec.model_id, port=int(args.port), timeout=600.0)
            records = records_in_bucket(split, env, "calibration")[:FOUNDATION_N]
            try:
                lab = OfficialLabEnv(env)
            except OfficialLabInfraError as exc:
                return {
                    "env": env,
                    "n": 0,
                    "success": "",
                    "n_success": 0,
                    "n_failure": 0,
                    "infra_failure_rate": 1.0,
                    "infra_error": str(exc),
                    "faked_success": False,
                }
            results = []
            try:
                for i, rec in enumerate(records):
                    write_status_live(
                        out / "STATUS_LIVE.json",
                        gpu=args.gpu,
                        stage="foundation",
                        batch_id=env,
                        completed_tasks=i,
                    )
                    results.append(
                        lab.run_episode(
                            rec.sample_index,
                            client,
                            state=None,
                            task_id=rec.task_id,
                            instruction_hint=rec.instruction,
                        )
                    )
            except OfficialLabInfraError as exc:
                return {
                    "env": env,
                    "n": 0,
                    "success": "",
                    "n_success": 0,
                    "n_failure": 0,
                    "infra_failure_rate": 1.0,
                    "infra_error": str(exc),
                    "faked_success": False,
                }
            finally:
                lab.close()
            stats = summarize_episodes(results)
            stats["env"] = env
            stats["no_learning"] = True
            return stats

        result = run_cell_once(cell, _job, resume=args.resume)
        payload = (result.get("payload") or {}) if isinstance(result, dict) else {}
        if payload.get("infra_error"):
            infra_errors.append(str(payload["infra_error"]))

    rows = write_foundation_csv(out, envs)
    gate = None
    try:
        numeric = [r for r in rows if str(r.get("success")) not in {"", "nan"}]
        if numeric:
            gate = classify_gate_f(numeric)
        elif infra_errors:
            gate = "CASE_F_INFRA_FAIL"
    except Exception:
        gate = None
    print(json.dumps({"csv": str(out / "R9_FOUNDATION.csv"), "gate_f": gate, "n_rows": len(rows)}))
    if infra_errors:
        raise OfficialLabInfraError("; ".join(infra_errors[:5]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
