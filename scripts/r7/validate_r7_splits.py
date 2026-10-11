#!/usr/bin/env python3
"""Validate frozen R7 splits: disjoint task ids / questions; answer-overlap report; sha256."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from expolicy.experiments.common import write_json
from expolicy.r7.constants import R7_ROOT


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def overlap(a: list[dict], b: list[dict], name: str) -> dict:
    ids_a, ids_b = {r["task_id"] for r in a}, {r["task_id"] for r in b}
    q_a = {str(r.get("question") or "").strip().lower() for r in a}
    q_b = {str(r.get("question") or "").strip().lower() for r in b}
    ans_a = {str(r.get("gold_answer") or "").strip().lower() for r in a if r.get("gold_answer")}
    ans_b = {str(r.get("gold_answer") or "").strip().lower() for r in b if r.get("gold_answer")}
    return {
        "pair": name,
        "task_id_overlap": len(ids_a & ids_b),
        "question_exact_overlap": len(q_a & q_b),
        "answer_overlap": len(ans_a & ans_b),
        "n_a": len(a),
        "n_b": len(b),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/r7_search_suite")
    p.add_argument("--require-task-id-disjoint", action="store_true")
    p.add_argument("--require-question-disjoint", action="store_true")
    p.add_argument("--require-answer-overlap-report", action="store_true")
    p.add_argument("--write-sha256", action="store_true")
    p.add_argument("--out", default=str(R7_ROOT / "audit" / "split"))
    args = p.parse_args()
    root = Path(args.root)
    split_path = root / "split.json"
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    keys = ["calibration", "acquisition_pool", "skill_validation", "future_probe"]
    buckets = {k: list(payload.get(k) or []) for k in keys}
    reports = []
    fail = False
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            rep = overlap(buckets[a], buckets[b], f"{a} vs {b}")
            reports.append(rep)
            if args.require_task_id_disjoint and rep["task_id_overlap"]:
                fail = True
            if args.require_question_disjoint and rep["question_exact_overlap"]:
                fail = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(split_path) if args.write_sha256 else ""
    write_json(out / "overlap.json", reports)
    if digest:
        (out / "SPLIT_HASH.txt").write_text(digest + "\n", encoding="utf-8")
        (root / "SPLIT_HASH.txt").write_text(digest + "\n", encoding="utf-8")
    actual = payload.get("actual_n") or {}
    md = ["# SPLIT_AUDIT\n", f"split.json sha256=`{digest}`\n", f"seed={payload.get('split_seed')}\n\n"]
    md.append("## actual n\n\n")
    md.append("```json\n" + json.dumps(actual, indent=2) + "\n```\n")
    md.append("\n## overlaps\n\n")
    md.append("| pair | task_id | question | answer | n_a | n_b |\n|---|---:|---:|---:|---:|---:|\n")
    for r in reports:
        md.append(
            f"| {r['pair']} | {r['task_id_overlap']} | {r['question_exact_overlap']} | {r['answer_overlap']} | {r['n_a']} | {r['n_b']} |\n"
        )
    md.append("\nAnswer overlap is reported, not forbidden (many QA items share short answers).\n")
    md.append(f"\nPASS={not fail}\n")
    (out / "SPLIT_AUDIT.md").write_text("".join(md), encoding="utf-8")
    (R7_ROOT / "audit" / "SPLIT_AUDIT.md").write_text("".join(md), encoding="utf-8")
    print("".join(md), flush=True)
    if fail:
        raise SystemExit("split isolation failed")


if __name__ == "__main__":
    main()
