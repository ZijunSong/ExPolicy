#!/usr/bin/env python3
"""Operator sanity: 2 batches x 8 experiences per env for A1/A2/A3."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from expolicy.r9_lab.audit.leakage import assert_future_not_in_update, future_ids
from expolicy.r9_lab.benchmark.official_env import OfficialLabEnv, OfficialLabInfraError
from expolicy.r9_lab.benchmark.split import load_split, records_in_bucket
from expolicy.r9_lab.constants import (
    A1_EPISODIC,
    A2_SEMANTIC,
    A3_SKILLOPT,
    ENVS,
    MATRIX_EXPERIENCES_PER_BATCH,
    OUTPUT_ROOT,
    SPLIT_SEED,
    VLLM_PORT,
    model_by_id,
)
from expolicy.r9_lab.io_util import read_jsonl
from expolicy.r9_lab.llm_client import VLLMClient, probe_vllm
from expolicy.r9_lab.manifest import load_manifest
from expolicy.r9_lab.matrix_runtime import (
    INJECTION_MARKERS,
    apply_action,
    as_experiences,
    fingerprint,
    make_validate_fn,
    record_to_item,
    run_items,
)
from expolicy.r9_lab.resume import run_cell_once
from expolicy.r9_lab.state.clone import clone_state
from expolicy.r9_lab.state.persistent_state import PersistentState, RETRIEVAL_TRACE
from expolicy.r9_lab.watchdog_util import append_recorded_pid, write_status_live

OPERATORS = (A1_EPISODIC, A2_SEMANTIC, A3_SKILLOPT)


def _blob(messages: list[dict[str, Any]]) -> str:
    return "\n".join(str(m.get("content") or "") for m in messages)


def _written_ids(state: PersistentState) -> list[str]:
    ids = list(state.committed_task_ids)
    ids.extend(r.task_id for r in state.episodic)
    for entry in state.semantic:
        ids.extend(entry.evidence_task_ids)
    return ids


def _run_env(
    *,
    env: str,
    out: Path,
    split: dict,
    client: Any,
    gpu: str | None,
    port: int,
) -> dict[str, Any]:
    val_recs = [record_to_item(r) for r in records_in_bucket(split, env, "assim_validation")]
    rng = random.Random(SPLIT_SEED + 5 + sum(ord(c) for c in env))
    pool = list(val_recs)
    rng.shuffle(pool)
    need = 2 * MATRIX_EXPERIENCES_PER_BATCH
    if len(pool) < need + 1:
        extra = [record_to_item(r) for r in records_in_bucket(split, env, "calibration")]
        for row in extra:
            if row["task_id"] not in {x["task_id"] for x in pool}:
                pool.append(row)
    used = pool[:need]
    follow = pool[need : need + 2] or pool[:1]
    batches = [
        used[i * MATRIX_EXPERIENCES_PER_BATCH : (i + 1) * MATRIX_EXPERIENCES_PER_BATCH]
        for i in range(2)
    ]
    status_path = out / "operator_sanity" / env / "STATUS_LIVE.json"
    s0_dir = out / "operator_sanity" / env / "S0"
    s0 = PersistentState.create(s0_dir)
    s0_hash = s0.content_hash()
    lab = OfficialLabEnv(env)
    checks: list[dict[str, Any]] = []
    failed: list[str] = []
    try:
        for bi, items in enumerate(batches):
            write_status_live(status_path, gpu=gpu, stage="operator_sanity_acquire", batch_id=f"{env}-S{bi}")
            acquire = run_items(
                lab,
                client,
                items,
                None,
                status_path=status_path,
                gpu=gpu,
                stage="operator_sanity_acquire",
                batch_id=f"{env}-S{bi}",
            )
            experiences = as_experiences(acquire, env=env, bucket="assim_validation")
            acquire_ids = [str(x["task_id"]) for x in experiences]
            n_success = sum(1 for x in experiences if x.get("success"))
            assert_future_not_in_update(acquire_ids, split)

            # Forced SkillOpt reject must not write.
            reject_dir = out / "operator_sanity" / env / f"S{bi}" / "skillopt_reject"
            reject_state = clone_state(s0, reject_dir)
            before_skill = reject_state.skill_markdown
            before_hash = reject_state.content_hash()
            reject_out = apply_action(
                A3_SKILLOPT,
                reject_state,
                experiences,
                client=client,
                validate_fn=lambda _skill: 0.0,
            )
            if reject_out.get("accepted") or reject_state.skill_markdown != before_skill or reject_state.content_hash() != before_hash:
                failed.append(f"{env}-S{bi}: SkillOpt reject wrote live skill")
            checks.append({"batch": f"S{bi}", "skillopt_reject_clean": reject_out.get("accepted") is False})

            clone_a = clone_state(s0, out / "operator_sanity" / env / f"S{bi}" / "clone_a")
            clone_b = clone_state(s0, out / "operator_sanity" / env / f"S{bi}" / "clone_b")
            apply_action(A2_SEMANTIC, clone_a, experiences, client=client)
            if PersistentState.load(clone_b.root_path()).content_hash() != s0_hash:
                failed.append(f"{env}-S{bi}: clone B polluted by clone A")
            if clone_a.content_hash() == s0_hash:
                failed.append(f"{env}-S{bi}: A2 did not change clone A")

            follow_item = follow[0]
            base_eps = run_items(lab, client, [follow_item], None)
            base_blob = _blob(base_eps[0].messages)

            for action in OPERATORS:
                branch = clone_state(s0, out / "operator_sanity" / env / f"S{bi}" / action)
                before_fp = fingerprint(branch)
                val_fn = make_validate_fn(
                    lab,
                    client,
                    pool[:4],
                    out / "operator_sanity" / env / f"S{bi}" / f"{action}_val",
                    status_path=status_path,
                    gpu=gpu,
                    batch_id=f"{env}-S{bi}:{action}",
                )
                assim = apply_action(action, branch, experiences, client=client, validate_fn=val_fn)
                after_fp = fingerprint(branch)
                changed = before_fp != after_fp
                if action == A1_EPISODIC:
                    if n_success > 0 and not changed:
                        failed.append(f"{env}-S{bi}: A1 had verified success but state files unchanged")
                    if n_success == 0 and changed:
                        failed.append(f"{env}-S{bi}: A1 wrote despite zero verified successes")
                elif action == A2_SEMANTIC:
                    if not changed:
                        failed.append(f"{env}-S{bi}: A2 semantic files unchanged")
                written = _written_ids(branch)
                try:
                    assert_future_not_in_update(written, split)
                    future_ok = True
                except AssertionError as exc:
                    failed.append(str(exc))
                    future_ok = False
                leaked_future = sorted(set(written) & future_ids(split))

                follow_on = run_items(lab, client, [follow_item], branch)
                on_blob = _blob(follow_on[0].messages)
                marker = INJECTION_MARKERS[action]
                read_ok = True
                if action == A1_EPISODIC and n_success > 0:
                    read_ok = marker in on_blob
                    if not read_ok:
                        failed.append(f"{env}-S{bi}: subsequent rollout did not read A1 memory")
                elif action == A2_SEMANTIC:
                    read_ok = marker in on_blob
                    if not read_ok:
                        failed.append(f"{env}-S{bi}: subsequent rollout did not read A2 memory")
                elif action == A3_SKILLOPT and assim.get("accepted"):
                    read_ok = marker in on_blob
                    if not read_ok:
                        failed.append(f"{env}-S{bi}: subsequent rollout did not read accepted skill")

                follow_off = run_items(lab, client, [follow_item], None)
                off_blob = _blob(follow_off[0].messages)
                disable_ok = marker not in off_blob
                if not disable_ok:
                    failed.append(f"{env}-S{bi}: disable {action} did not return to base prompt path")
                base_ok = marker not in base_blob
                if not base_ok:
                    failed.append(f"{env}-S{bi}: base path already contained {action} marker")

                traces = read_jsonl(branch.root_path() / RETRIEVAL_TRACE)
                hit_ids = []
                for row in traces:
                    hit_ids.extend(str(x) for x in (row.get("hit_task_ids") or []))
                training_hits = [h for h in hit_ids if h]
                retrieval_ok = all(h in acquire_ids for h in training_hits)
                if not retrieval_ok:
                    failed.append(f"{env}-S{bi}: retrieval hit non-training ids {training_hits}")

                checks.append(
                    {
                        "batch": f"S{bi}",
                        "action": action,
                        "changed": changed,
                        "assim": {k: assim.get(k) for k in ("action", "changed", "accepted", "n", "reason") if k in assim},
                        "n_success_in_batch": n_success,
                        "read_by_followup": read_ok,
                        "disable_returns_base": disable_ok,
                        "future_ok": future_ok,
                        "leaked_future": leaked_future,
                        "retrieval_training_side": retrieval_ok,
                        "hit_task_ids": training_hits,
                    }
                )
    finally:
        lab.close()

    status = "OPERATOR_SANITY_PASS" if not failed else "OPERATOR_SANITY_FAIL"
    return {
        "env": env,
        "status": status,
        "failed": failed,
        "n_batches": len(batches),
        "batch_task_ids": [[x["task_id"] for x in b] for b in batches],
        "follow_task_ids": [x["task_id"] for x in follow],
        "checks": checks,
        "port": port,
        "used_future_test": False,
    }


def write_sanity_md(out: Path, env_reports: list[dict[str, Any]]) -> Path:
    md = out / "audit" / "OPERATOR_SANITY.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    failed = [f for r in env_reports for f in (r.get("failed") or [])]
    status = "OPERATOR_SANITY_PASS" if env_reports and not failed and len(env_reports) >= 1 else "OPERATOR_SANITY_FAIL"
    if any(r.get("status") == "OPERATOR_SANITY_FAIL" for r in env_reports):
        status = "OPERATOR_SANITY_FAIL"
    lines = [
        "# OPERATOR_SANITY",
        "",
        f"**status:** {status}",
        "",
        "Checks: state files change; subsequent rollout reads new state; disable returns to base;",
        "clone A does not pollute clone B; future_test ids never enter updates; SkillOpt reject does not write;",
        "retrieval traces hit training-side experiences only.",
        "",
    ]
    for report in env_reports:
        lines.append(f"## {report.get('env')}")
        lines.append("")
        lines.append(f"- env_status: `{report.get('status')}`")
        lines.append(f"- batches: `{report.get('batch_task_ids')}`")
        if report.get("failed"):
            lines.append("- failures:")
            for item in report["failed"]:
                lines.append(f"  - {item}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(report.get("checks") or [], indent=2, ensure_ascii=False)[:12000])
        lines.append("```")
        lines.append("")
    if failed:
        lines.append("## Failures")
        lines.append("")
        for item in failed:
            lines.append(f"- {item}")
        lines.append("")
    md.write_text("\n".join(lines), encoding="utf-8")
    return md


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--gpu", default=None)
    p.add_argument("--port", type=int, default=VLLM_PORT)
    p.add_argument("--env", action="append", dest="envs")
    p.add_argument("--allow-synthetic", action="store_true")
    args = p.parse_args(argv)
    out = args.output_root
    envs = args.envs or list(ENVS)
    append_recorded_pid(out / "jobs", os.getpid(), " ".join(["05_operator_sanity", *envs]))
    md = out / "audit" / "OPERATOR_SANITY.md"
    md.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:

        def _dry() -> dict:
            md.write_text(
                "# OPERATOR_SANITY (dry-run placeholder)\n\n"
                f"envs={list(envs)}\noperators={list(OPERATORS)}\n"
                "Real sanity requires GPU/LAB. Unit tests cover isolation invariants.\n",
                encoding="utf-8",
            )
            return {"status": "DRY_RUN", "path": str(md)}

        print(json.dumps(run_cell_once(out / "operator_sanity", _dry, resume=args.resume), default=str))
        return 0

    man = load_manifest(out / "model_manifest.json")
    if not man.get("selected_model_id"):
        raise RuntimeError("05_operator_sanity requires frozen selected actor")
    spec = model_by_id(str(man["selected_model_id"]))
    split = load_split(out / "split.json")
    if split.get("synthetic") and not args.allow_synthetic:
        raise RuntimeError("Refusing synthetic split.json unless --allow-synthetic")
    probe_vllm(int(args.port))
    client = VLLMClient.from_model_id(spec.model_id, port=int(args.port), timeout=600.0)
    reports: list[dict] = []
    for env in envs:
        cell = out / "operator_sanity" / env

        def _env_job(env=env, cell=cell) -> dict:
            write_status_live(
                cell / "STATUS_LIVE.json",
                gpu=args.gpu,
                stage="operator_sanity",
                batch_id=env,
            )
            try:
                return _run_env(
                    env=env,
                    out=out,
                    split=split,
                    client=client,
                    gpu=args.gpu,
                    port=int(args.port),
                )
            except OfficialLabInfraError as exc:
                return {
                    "env": env,
                    "status": "OPERATOR_SANITY_FAIL",
                    "failed": [f"infra: {exc}"],
                    "checks": [],
                }

        result = run_cell_once(cell, _env_job, resume=args.resume)
        payload = result.get("payload") if isinstance(result, dict) and not result.get("skipped") else None
        if payload is None and (cell / "DONE.json").exists():
            from expolicy.r9_lab.io_util import read_json

            raw = read_json(cell / "DONE.json")
            payload = raw.get("payload") or raw
        if payload:
            reports.append(payload)
    md_path = write_sanity_md(out, reports)
    status = "OPERATOR_SANITY_PASS"
    if any(r.get("status") != "OPERATOR_SANITY_PASS" for r in reports) or len(reports) < len(envs):
        status = "OPERATOR_SANITY_FAIL"
    print(json.dumps({"status": status, "path": str(md_path), "envs": [r.get("env") for r in reports]}, default=str))
    if status == "OPERATOR_SANITY_FAIL":
        raise SystemExit("OPERATOR_SANITY_FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
