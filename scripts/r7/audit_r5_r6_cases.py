#!/usr/bin/env python3
"""Human-readable R5.3 / R5.4 / R6 casebook. Real artifacts only."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from expolicy.experiments.common import ROOT, write_json
from expolicy.r7.constants import R7_ROOT


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_split_index(r53: Path) -> dict[str, dict]:
    path = r53 / "splits" / "split.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    idx = {}
    for key in ("calibration", "acquisition_pool", "skill_validation", "future_probe", "bamboogle"):
        for rec in payload.get(key) or []:
            idx[rec["task_id"]] = rec
    return idx


def yaml_case(rec: dict) -> str:
    keys = [
        "question",
        "dataset",
        "predicted_capability",
        "base_answer",
        "base_correct",
        "acquisition_reason",
        "retrieved_docs",
        "skill_before",
        "skill_after",
        "skill_interventions",
        "model_answer_after_skill",
        "correct_after",
        "what_changed",
        "failure_origin",
    ]
    lines = ["---"]
    for k in keys:
        v = rec.get(k)
        if isinstance(v, (list, dict)):
            v = json.dumps(v, ensure_ascii=False)[:400]
        if v is None:
            v = ""
        s = str(v).replace("\n", " ").strip()
        if len(s) > 400:
            s = s[:400] + "..."
        lines.append(f"{k}: {s}")
    lines.append("---")
    return "\n".join(lines)


def sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:n]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--r5-3", dest="r53", default="outputs/r5_3_external_generalization")
    p.add_argument("--r5-4", dest="r54", default="outputs/r5_4_rasa")
    p.add_argument("--r6", default="outputs/r6_ava")
    p.add_argument("--per-condition-cases", type=int, default=30)
    p.add_argument("--out", default=str(R7_ROOT / "audit" / "cases"))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260815)
    r53 = ROOT / args.r53
    r54 = ROOT / args.r54
    r6 = ROOT / args.r6
    idx = load_split_index(r53)
    n = args.per_condition_cases

    buckets: dict[str, list[dict]] = defaultdict(list)

    # R5.3 frontier C1 errors
    sel = read_jsonl(r53 / "frontier" / "seed_0" / "selected_experiences.jsonl")
    for rec in sel:
        meta = idx.get(rec.get("task_id") or "", {})
        if rec.get("capability") == "bridge_decomposition" and not rec.get("success"):
            buckets["r53_frontier_c1_error"].append(
                {
                    "question": meta.get("question"),
                    "dataset": rec.get("dataset") or meta.get("dataset"),
                    "predicted_capability": rec.get("capability"),
                    "base_answer": rec.get("predicted"),
                    "base_correct": False,
                    "acquisition_reason": "legacy_frontier_score; 48/48 C1 on public data",
                    "retrieved_docs": "",
                    "skill_before": "inactive (count<8 early; then C1 only)",
                    "skill_after": "count_only C1 primitive",
                    "skill_interventions": "replacement search/open/heuristic_answer",
                    "model_answer_after_skill": rec.get("predicted"),
                    "correct_after": rec.get("success"),
                    "what_changed": "frontier allocated C1 only; skill later replaces model on C1 future items",
                    "failure_origin": "acquisition allocation concentrated on C1; C2 (the only large skill lift) starved",
                    "task_id": rec.get("task_id"),
                }
            )

    # C2 successes vs C1: compare R5.3 methods via selected + known C2 lift from R6 masks
    ace = read_jsonl(r53 / "ace_skill_prioritized" / "seed_0" / "selected_experiences.jsonl")
    pas = read_jsonl(r53 / "passive" / "seed_0" / "selected_experiences.jsonl")
    for rec in ace + pas:
        meta = idx.get(rec.get("task_id") or "", {})
        if rec.get("capability") == "comparison_alignment" and rec.get("success"):
            buckets["c2_skill_success_proxy"].append(
                {
                    "question": meta.get("question"),
                    "dataset": rec.get("dataset") or meta.get("dataset"),
                    "predicted_capability": rec.get("capability"),
                    "base_answer": rec.get("predicted"),
                    "base_correct": True,
                    "acquisition_reason": "ace_or_passive sampled C2",
                    "retrieved_docs": "",
                    "skill_before": "none",
                    "skill_after": "count_only C2",
                    "skill_interventions": "comparison procedure",
                    "model_answer_after_skill": rec.get("predicted"),
                    "correct_after": True,
                    "what_changed": "C2 experiences acquired; CountOnly can activate comparison skill",
                    "failure_origin": "n/a (success)",
                    "task_id": rec.get("task_id"),
                }
            )
        if rec.get("capability") == "comparison_alignment" and not rec.get("success"):
            buckets["skill_activated_still_fail_proxy"].append(
                {
                    "question": meta.get("question"),
                    "dataset": rec.get("dataset") or meta.get("dataset"),
                    "predicted_capability": rec.get("capability"),
                    "base_answer": rec.get("predicted"),
                    "base_correct": False,
                    "acquisition_reason": "ace_or_passive C2 sample",
                    "retrieved_docs": "",
                    "skill_before": "none",
                    "skill_after": "count_only C2 (if K reached)",
                    "skill_interventions": "comparison procedure / heuristic span",
                    "model_answer_after_skill": rec.get("predicted"),
                    "correct_after": False,
                    "what_changed": "skill procedure ran or base failed before activation",
                    "failure_origin": "skill itself not always repairable; heuristic span extraction",
                    "task_id": rec.get("task_id"),
                }
            )

    fr_ids = {r.get("task_id") for r in sel}
    ace_ids = {r.get("task_id") for r in ace}
    pas_ids = {r.get("task_id") for r in pas}
    for rec in ace + pas:
        tid = rec.get("task_id")
        if tid in fr_ids:
            continue
        meta = idx.get(tid or "", {})
        if rec.get("success"):
            buckets["passive_ace_success_frontier_missed"].append(
                {
                    "question": meta.get("question"),
                    "dataset": rec.get("dataset") or meta.get("dataset"),
                    "predicted_capability": rec.get("capability"),
                    "base_answer": rec.get("predicted"),
                    "base_correct": True,
                    "acquisition_reason": "sampled by ace/passive, not by frontier seed0",
                    "retrieved_docs": "",
                    "skill_before": "n/a",
                    "skill_after": "n/a",
                    "skill_interventions": "",
                    "model_answer_after_skill": rec.get("predicted"),
                    "correct_after": True,
                    "what_changed": "frontier never acquired this task/cap",
                    "failure_origin": "acquisition allocation (frontier C1 tunnel)",
                    "task_id": tid,
                }
            )

    # R5.4 C3 UNRESOLVED
    post = r54 / "R5_4_POSTERIORS.csv"
    if post.exists():
        import csv

        with post.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("capability") == "long_chain_search" and row.get("state") == "UNRESOLVED" and row.get("budget") == "48":
                    buckets["r54_c3_unresolved"].append(
                        {
                            "question": "(capability-level posterior, not a single question)",
                            "dataset": "mixed",
                            "predicted_capability": "long_chain_search",
                            "base_answer": "",
                            "base_correct": False,
                            "acquisition_reason": f"rasa condition={row.get('condition')} seed={row.get('seed')}",
                            "retrieved_docs": "",
                            "skill_before": "UNRESOLVED",
                            "skill_after": "still UNRESOLVED; skill not ACTIVE",
                            "skill_interventions": f"repair_count={row.get('repair_count')} n_tested_base_failures={row.get('n_tested_base_failures')}",
                            "model_answer_after_skill": "",
                            "correct_after": False,
                            "what_changed": "C3 stayed UNRESOLVED so CountOnly-style activation never fired under SRT",
                            "failure_origin": "router/posterior: replay p too low vs PARK; skill not installed",
                            "task_id": f"posterior:{row.get('condition')}:seed{row.get('seed')}:C3",
                        }
                    )

    # existing R6 case table
    for rec in read_jsonl(r6 / "case_analysis" / "CASE_TABLE.jsonl"):
        origin = "skill_not_uniformly_useless_on_c3"
        if rec.get("router_cap") == "long_chain_search" and rec.get("base_correct") is False:
            origin = "c3_base_fail"
        buckets["r54_case_table"].append(
            {
                "question": rec.get("question"),
                "dataset": rec.get("dataset"),
                "predicted_capability": rec.get("router_cap"),
                "base_answer": "",
                "base_correct": rec.get("base_correct"),
                "acquisition_reason": "r5.4 rasa / r6 case table",
                "retrieved_docs": "",
                "skill_before": "",
                "skill_after": "",
                "skill_interventions": rec.get("base_trace", "")[:200],
                "model_answer_after_skill": "",
                "correct_after": rec.get("skill_correct"),
                "what_changed": "see R6 CASE_REPORT",
                "failure_origin": origin,
                "task_id": rec.get("task_id"),
            }
        )

    # R6 AVA seed0 C1 occupancy
    ava_sel = read_jsonl(r6 / "per_seed" / "exact_ava" / "seed_0" / "selected_experiences.jsonl")
    c1 = [r for r in ava_sel if r.get("capability") == "bridge_decomposition" or r.get("predicted_capability") == "bridge_decomposition"]
    for rec in c1:
        meta = idx.get(rec.get("task_id") or "", {})
        buckets["r6_ava_seed0_c1"].append(
            {
                "question": meta.get("question"),
                "dataset": rec.get("dataset") or meta.get("dataset"),
                "predicted_capability": rec.get("capability") or rec.get("predicted_capability"),
                "base_answer": rec.get("predicted"),
                "base_correct": rec.get("success"),
                "acquisition_reason": "exact_ava seed0; CountOnly still collapses to mask",
                "retrieved_docs": "",
                "skill_before": "mask lookup",
                "skill_after": "mask lookup",
                "skill_interventions": "old replacement skill",
                "model_answer_after_skill": rec.get("predicted"),
                "correct_after": rec.get("success"),
                "what_changed": "oracle AVA cannot beat passive because backend is mask lookup",
                "failure_origin": "assimilation backend (CountOnly), not value estimator",
                "task_id": rec.get("task_id"),
            }
        )

    chosen: dict[str, list[dict]] = {}
    for key, rows in buckets.items():
        chosen[key] = sample(rows, n, rng)
        write_json(out / f"{key}.json", chosen[key])

    md = ["# R7_CASEBOOK\n", "Real R5.3 / R5.4 / R6 artifacts. No synthetic cases.\n"]
    mapping = [
        ("r53_frontier_c1_error", "R5.3 frontier errors concentrated on C1"),
        ("r54_c3_unresolved", "R5.4 C3 UNRESOLVED / rare repair"),
        ("r6_ava_seed0_c1", "R6 AVA seed0 remaining C1-heavy / mask-collapse"),
        ("c2_skill_success_proxy", "C2 skill-adjacent successes (Ace/passive acquired C2)"),
        ("skill_activated_still_fail_proxy", "C2/C3 acquired but still failed"),
        ("passive_ace_success_frontier_missed", "Ace/passive acquired successes frontier missed"),
        ("r54_case_table", "R6 CASE_TABLE remainder (C3 base-fail / repair sparsity)"),
    ]
    for key, title in mapping:
        rows = chosen.get(key) or []
        md.append(f"\n## {title}\n\n n_sampled={len(rows)} from_bucket={len(buckets.get(key) or [])}\n")
        for rec in rows:
            md.append(yaml_case(rec))
            md.append("\n")
    md.append(
        "\n## Failure-origin summary\n"
        "- R5.3 frontier: **acquisition allocation** (48/48 C1) plus **skill executor replaces model**.\n"
        "- R5.4 C3: **router/posterior** keeps cap UNRESOLVED; frozen C3 primitive never ACTIVE under SRT.\n"
        "- R6 AVA: **skill itself is a mask lookup**; exact utility has no content-sensitive headroom.\n"
        "- C2 is the only large CountOnly lift; that lift is **replacement-executor span extraction**, not model generalization.\n"
    )
    casebook = R7_ROOT / "audit" / "R7_CASEBOOK.md"
    casebook.write_text("".join(md), encoding="utf-8")
    print(f"wrote {casebook} buckets={ {k: len(v) for k,v in chosen.items()} }", flush=True)


if __name__ == "__main__":
    main()
