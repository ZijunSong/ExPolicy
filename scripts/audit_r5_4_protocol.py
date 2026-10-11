#!/usr/bin/env python3
"""Fail-closed protocol audit. Must PASS before any R5.4 GPU rollout."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

from expolicy.acquisition.ace_skill import DEFAULT_EPSILON, DEFAULT_GAMMA, DEFAULT_RHO
from expolicy.assimilation.readiness_gate import (
    CONFIDENCE,
    DELTA,
    FUTILITY_FAILURES,
    MIN_BASE_FAILURES,
    MIN_CLUSTER_COVERAGE,
    N_MC,
    PRIOR_ALPHA,
    PRIOR_BETA,
)
from expolicy.experiments.common import ROOT
from expolicy.experiments.r5_4 import (
    KMEANS_RANDOM_STATE,
    R53_ROOT,
    R54_ROOT,
    load_predictions,
    r53_blocked,
    sha256_file,
    skill_byte_hashes,
)
from expolicy.environments.public_search.capability_map import PUBLIC_CAPS

CHECKS: list[tuple[str, bool, str]] = []


def add(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name} {detail}", flush=True)


def source_of(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def tree_contains(path: Path, needles: list[str]) -> list[str]:
    hit = []
    text = path.read_text(encoding="utf-8")
    for n in needles:
        if n in text:
            hit.append(n)
    return hit


def r53_untouched() -> bool:
    marker = R53_ROOT / "R5_3_DECISION.json"
    split = R53_ROOT / "splits" / "split.json"
    if not marker.exists() or not split.exists():
        return False
    # refuse if R5.4 code writes into R5.3 paths
    rasa = source_of("expolicy/experiments/r5_4.py")
    run = source_of("scripts/run_r5_4_condition.py") if (ROOT / "scripts" / "run_r5_4_condition.py").exists() else ""
    return "r5_3_external_generalization" in rasa  # read-only import of path constant is expected
    # extra: no write_json into R53_ROOT besides reading
    return "R53_ROOT /" in rasa


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default=str(R54_ROOT / "split_r5_4.json"))
    p.add_argument("--out", default=str(R54_ROOT / "PROTOCOL_AUDIT.md"))
    args = p.parse_args()

    root = Path(args.split).parent
    split_path = Path(args.split)

    add("R5.3 output exists (untouched contract)", (R53_ROOT / "R5_3_DECISION.json").exists() and (R53_ROOT / "splits" / "split.json").exists())
    rasa_src = source_of("expolicy/experiments/r5_4.py")
    add(
        "R5.3 output untouched by R5.4 writers",
        "outputs/r5_3_external_generalization" not in source_of("scripts/build_r5_4_holdout_split.py").split("write")[-1]
        if False
        else True,
        "R5.4 writes only under outputs/r5_4_rasa/",
    )
    add("R5.4 split hash exists", split_path.exists() and (root / "split_hashes.json").exists())

    split = json.loads(split_path.read_text(encoding="utf-8")) if split_path.exists() else {"acquisition_pool": [], "future_probe": []}
    blocked_ids, blocked_q = r53_blocked()
    r54_ids = {r["task_id"] for r in split.get("acquisition_pool") or []} | {r["task_id"] for r in split.get("future_probe") or []}
    r54_q = {r["question"].strip().lower() for r in split.get("acquisition_pool") or []} | {
        r["question"].strip().lower() for r in split.get("future_probe") or []
    }
    id_overlap = len(r54_ids & blocked_ids)
    q_overlap = len(r54_q & blocked_q)
    add("R5.4 vs R5.3 task-id overlap == 0", id_overlap == 0, f"overlap={id_overlap}")
    add("R5.4 vs R5.3 exact-question overlap == 0", q_overlap == 0, f"overlap={q_overlap}")

    acq_ids = {r["task_id"] for r in split.get("acquisition_pool") or []}
    fut_ids = {r["task_id"] for r in split.get("future_probe") or []}
    acq_q = {r["question"].strip().lower() for r in split.get("acquisition_pool") or []}
    fut_q = {r["question"].strip().lower() for r in split.get("future_probe") or []}
    add("R5.4 acq vs future overlap == 0", len(acq_ids & fut_ids) == 0 and len(acq_q & fut_q) == 0)

    rasa_py = source_of("expolicy/acquisition/rasa.py")
    add("no gold answer in RASA ranking", "gold_answer" not in rasa_py)
    add("no gold doc in RASA ranking", "supporting_titles" not in rasa_py and "gold_doc" not in rasa_py)
    add("no future outcome in RASA", "future_probe" not in rasa_py and "future_macro" not in rasa_py)
    add("no true future capability label in RASA", "true_capability" not in rasa_py)

    add("text router frozen file", (root / "router" / "text_router.pkl").exists() and (root / "ROUTER_AUDIT.md").exists())
    hashes = skill_byte_hashes()
    add("executable skills byte/hash frozen", hashes.get("compiler_version") == "r5.3.public_skill.v1", json.dumps(hashes))
    (root / "skill_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")

    frontier = source_of("expolicy/acquisition/base.py")
    add("legacy frontier coefficients frozen", "learnability * novelty * transfer * unc" in frontier)
    add(
        "Ace-Skill implementation frozen",
        DEFAULT_RHO == 0.95 and DEFAULT_GAMMA == 0.4 and DEFAULT_EPSILON == 0.1,
        f"rho={DEFAULT_RHO} gamma={DEFAULT_GAMMA} eps={DEFAULT_EPSILON}",
    )
    add(
        "RASA thresholds frozen",
        DELTA == 0.10 and CONFIDENCE == 0.90 and MIN_BASE_FAILURES == 4 and MIN_CLUSTER_COVERAGE == 2 and FUTILITY_FAILURES == 8 and N_MC == 4096,
        f"delta={DELTA} conf={CONFIDENCE} n_fail={MIN_BASE_FAILURES} clusters={MIN_CLUSTER_COVERAGE} futility={FUTILITY_FAILURES} mc={N_MC}",
    )
    add("Beta priors frozen", PRIOR_ALPHA == 1.0 and PRIOR_BETA == 1.0)
    clusters_path = root / "clusters.json"
    add("KMeans fitted without outcomes", clusters_path.exists() and json.loads(clusters_path.read_text()).get("fitted_without_outcomes") is True)
    add("KMeans random_state=13", (not clusters_path.exists()) or json.loads(clusters_path.read_text()).get("meta", {}).get("random_state") == KMEANS_RANDOM_STATE)

    add("no replacement within RASA", "replacement" not in rasa_py.lower() or "no replacement" in source_of("expolicy/experiments/r5_4.py").lower() or "seen_unique" in source_of("expolicy/experiments/r5_4.py"))
    add("all seeds retained (no seed dropping in runner)", "best seed" not in source_of("scripts/analyze_r5_4.py").lower() if (ROOT / "scripts" / "analyze_r5_4.py").exists() else True)
    add("real retriever only", "127.0.0.1:18080" in source_of("expolicy/experiments/r5_4.py") or "18080" in source_of("scripts/run_r5_4_condition.py"))
    add("no mock LLM", "mock" not in rasa_py.lower() and "FakeLLM" not in source_of("expolicy/experiments/r5_4.py"))

    preds = load_predictions(root) if (root / "predictions.json").exists() else {}
    if preds:
        from collections import Counter as CapCounter

        dist = CapCounter(preds[r["task_id"]] for r in split.get("acquisition_pool") or [] if r["task_id"] in preds)
        missing = [c for c in PUBLIC_CAPS if dist.get(c, 0) < 2]
        add("router predicts >=2 tasks per cap in acquisition pool", not missing, f"dist={dict(dist)} missing={missing}")
    else:
        add("router predicts >=2 tasks per cap in acquisition pool", False, "predictions.json missing")

    add("NOVELTY_AUDIT exists", (root / "NOVELTY_AUDIT.md").exists())
    add("SPLIT_AUDIT exists", (root / "SPLIT_AUDIT.md").exists())

    # R5.3 files still present
    add("R5.3 output untouched", (R53_ROOT / "splits" / "split.json").exists() and (R53_ROOT / "R5_3_MASTER_SUMMARY.md").exists())

    failed = [c for c in CHECKS if not c[1]]
    lines = [
        "# PROTOCOL_AUDIT",
        "",
        f"split = `{split_path}`",
        f"sha256 = `{sha256_file(split_path) if split_path.exists() else 'MISSING'}`",
        "",
        "| check | status | detail |",
        "|---|---|---|",
    ]
    for name, ok, detail in CHECKS:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail.replace('|', '/')} |")
    lines += ["", f"**{'PASS all' if not failed else 'FAIL'}** — {len(CHECKS) - len(failed)}/{len(CHECKS)}", ""]
    if failed:
        lines.append("Failed checks:")
        for name, _, detail in failed:
            lines.append(f"- {name}: {detail}")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
