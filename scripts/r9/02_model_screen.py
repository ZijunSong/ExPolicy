#!/usr/bin/env python3
"""Model-screen: 24 no-learning episodes per (model, env). Writes R9_MODEL_SCREEN.csv."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

from expolicy.r9_lab.benchmark.official_env import (
    OfficialLabEnv,
    OfficialLabInfraError,
    summarize_episodes,
)
from expolicy.r9_lab.benchmark.split import load_split, records_in_bucket
from expolicy.r9_lab.constants import (
    ENVS,
    MODEL_LADDER,
    MODEL_SCREEN_N,
    OUTPUT_ROOT,
    vllm_port_for_model,
)
from expolicy.r9_lab.llm_client import VLLMClient, probe_vllm
from expolicy.r9_lab.resume import run_cell_once
from expolicy.r9_lab.watchdog_util import write_status_live

CSV_FIELDS = [
    "model_id",
    "hf_repo",
    "env",
    "n",
    "success",
    "parser_valid_rate",
    "infra_failure_rate",
    "mean_steps",
    "invalid_action_rate",
    "tokens",
    "reasoning_tokens",
    "wall_time",
]


def _stats_row(spec, env: str, stats: dict[str, Any]) -> dict[str, Any]:
    row = {k: "" for k in CSV_FIELDS}
    row.update(
        {
            "model_id": spec.model_id,
            "hf_repo": spec.hf_repo,
            "env": env,
            "n": stats.get("n", ""),
            "success": stats.get("success", ""),
            "parser_valid_rate": stats.get("parser_valid_rate", ""),
            "infra_failure_rate": stats.get("infra_failure_rate", ""),
            "mean_steps": stats.get("mean_steps", ""),
            "invalid_action_rate": stats.get("invalid_action_rate", ""),
            "tokens": stats.get("tokens", ""),
            "reasoning_tokens": stats.get("reasoning_tokens", ""),
            "wall_time": stats.get("wall_time", ""),
        }
    )
    return row


def run_screen_cell(
    *,
    spec,
    env: str,
    records,
    port: int,
    gpu: str | None,
    output_root: Path,
    env_ctor: Callable[[str], Any] | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    if client is None:
        try:
            probe_vllm(port)
        except Exception as exc:  # noqa: BLE001
            raise OfficialLabInfraError(str(exc)) from exc
        client = VLLMClient.from_model_id(spec.model_id, port=port, timeout=600.0)
    ctor = env_ctor or OfficialLabEnv
    lab = ctor(env)
    results = []
    try:
        for i, rec in enumerate(records):
            write_status_live(
                output_root / "STATUS_LIVE.json",
                gpu=gpu,
                stage="model_screen",
                batch_id=f"{spec.model_id}:{env}",
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
    finally:
        close = getattr(lab, "close", None)
        if callable(close):
            close()
    stats = summarize_episodes(results)
    stats["model_id"] = spec.model_id
    stats["env"] = env
    stats["used_future_test"] = False
    stats["no_learning"] = True
    return stats


def write_csv_from_cells(out: Path, specs, envs: list[str]) -> int:
    rows = []
    for spec in specs:
        for env in envs:
            done = out / "model_screen" / spec.model_id / env / "DONE.json"
            if not done.exists():
                continue
            payload = json.loads(done.read_text(encoding="utf-8"))
            body = payload.get("payload") or payload
            rows.append(_stats_row(spec, env, body))
    csv_path = out / "R9_MODEL_SCREEN.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--model-id", action="append", dest="model_ids")
    p.add_argument("--env", action="append", dest="envs")
    p.add_argument("--gpu", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--allow-synthetic", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Write empty placeholder rows without GPU")
    args = p.parse_args(argv)
    out = args.output_root
    models = [m for m in MODEL_LADDER if (not args.model_ids or m.model_id in args.model_ids)]
    envs = args.envs or list(ENVS)

    split = None
    if not args.dry_run:
        split_path = out / "split.json"
        if not split_path.exists():
            raise FileNotFoundError(f"split.json missing: {split_path}")
        split = load_split(split_path)
        if split.get("synthetic") and not args.allow_synthetic:
            raise RuntimeError("Refusing synthetic split.json unless --allow-synthetic")

    infra_errors: list[str] = []
    for spec in models:
        port = int(args.port if args.port is not None else vllm_port_for_model(spec.model_id))
        for env in envs:
            cell = out / "model_screen" / spec.model_id / env

            def _job(spec=spec, env=env, port=port) -> dict:
                if args.dry_run:
                    return {
                        "model_id": spec.model_id,
                        "env": env,
                        "n": MODEL_SCREEN_N,
                        "success": "nan",
                        "parser_valid_rate": "nan",
                        "infra_failure_rate": "nan",
                        "dry_run": True,
                    }
                records = records_in_bucket(split, env, "model_screen")[:MODEL_SCREEN_N]
                try:
                    return run_screen_cell(
                        spec=spec,
                        env=env,
                        records=records,
                        port=port,
                        gpu=args.gpu,
                        output_root=out,
                    )
                except OfficialLabInfraError as exc:
                    return {
                        "model_id": spec.model_id,
                        "env": env,
                        "n": 0,
                        "success": "",
                        "parser_valid_rate": "",
                        "infra_failure_rate": 1.0,
                        "infra_error": str(exc),
                        "faked_success": False,
                    }

            result = run_cell_once(cell, _job, resume=args.resume)
            payload = (result.get("payload") or {}) if isinstance(result, dict) else {}
            if payload.get("infra_error"):
                infra_errors.append(str(payload["infra_error"]))

    # Always rewrite the full CSV from every DONE cell so per-env launches
    # cannot wipe sibling environment rows.
    n_rows = write_csv_from_cells(out, MODEL_LADDER, list(ENVS))
    print(json.dumps({"csv": str(out / "R9_MODEL_SCREEN.csv"), "n_rows": n_rows}))
    if infra_errors:
        raise OfficialLabInfraError("; ".join(infra_errors[:5]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
