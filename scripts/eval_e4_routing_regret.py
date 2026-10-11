#!/usr/bin/env python3
"""E4 diagnostic: routing regret on a collected E3 matrix. No new GPU rollouts."""

from __future__ import annotations

import json
from pathlib import Path

from expolicy.assimilation.abstractor import ExperienceAbstractor
from expolicy.assimilation.router import HeuristicRouter
from expolicy.assimilation.verifier import ExperienceVerifier
from expolicy.core.experience import Experience
from expolicy.experiments.common import ROOT, load_world, mean, write_json

E0 = ROOT / "outputs" / "e0_audit"
E3B = ROOT / "outputs" / "e3b_carrier_matrix"
OUT = ROOT / "outputs" / "e4_routing_regret"
MEMORY_CARRIERS = ["episodic_memory", "semantic_memory", "skill"]


def _load_exps() -> dict[str, Experience]:
    rows: dict[str, Experience] = {}
    for stage in range(5):
        path = E0 / f"stage_{stage}" / "experiences.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                exp = Experience.model_validate(json.loads(line))
                rows[exp.experience_id] = exp
    return rows


def _utility(row: dict, carrier: str) -> float | None:
    cell = row.get(carrier)
    if isinstance(cell, dict) and "utility" in cell:
        return float(cell["utility"])
    return None


def main() -> None:
    world = load_world()
    exps = _load_exps()
    rows = [json.loads(l) for l in (E3B / "matrix.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    verifier = ExperienceVerifier(world)
    abstractor = ExperienceAbstractor()
    router = HeuristicRouter()
    records = []
    for row in rows:
        exp = exps[row["experience_id"]]
        ver = verifier.verify(exp)
        abs_ = abstractor.abstract(exp, ver)
        decision = router.route(exp, ver, abs_)
        chosen = decision.carrier.value
        utils = {c: _utility(row, c) for c in MEMORY_CARRIERS if _utility(row, c) is not None}
        oracle = max(utils, key=utils.get) if utils else None
        heuristic_u = utils.get(chosen)
        if heuristic_u is None:
            # harness/weight/discard not in this matrix → map to worst available (explicit, not silent win)
            heuristic_u = min(utils.values()) if utils else 0.0
            mapped = True
        else:
            mapped = False
        rec = {
            "experience_id": exp.experience_id,
            "task_id": exp.task_id,
            "capability": exp.capability,
            "success": exp.success,
            "heuristic_carrier": chosen,
            "heuristic_rationale": decision.rationale,
            "heuristic_mapped_from_unavailable": mapped,
            "oracle_carrier": oracle,
            "utilities": utils,
            "oracle_u": utils.get(oracle) if oracle else None,
            "heuristic_u": heuristic_u,
            "fixed_episodic_u": utils.get("episodic_memory"),
            "fixed_semantic_u": utils.get("semantic_memory"),
            "fixed_skill_u": utils.get("skill"),
        }
        rec["regret_heuristic"] = rec["oracle_u"] - rec["heuristic_u"]
        rec["regret_fixed_episodic"] = rec["oracle_u"] - rec["fixed_episodic_u"]
        rec["regret_fixed_semantic"] = rec["oracle_u"] - rec["fixed_semantic_u"]
        rec["regret_fixed_skill"] = rec["oracle_u"] - rec["fixed_skill_u"]
        records.append(rec)

    def _mean_regret(key: str) -> float:
        return mean(r[key] for r in records)

    # random: expected U = mean of available carriers, regret vs oracle
    random_regrets = []
    for r in records:
        eu = mean(r["utilities"].values())
        random_regrets.append(r["oracle_u"] - eu)

    summary = {
        "n": len(records),
        "matrix": str(E3B),
        "mean_oracle_u": mean(r["oracle_u"] for r in records),
        "mean_heuristic_u": mean(r["heuristic_u"] for r in records),
        "mean_fixed_episodic_u": mean(r["fixed_episodic_u"] for r in records),
        "mean_fixed_semantic_u": mean(r["fixed_semantic_u"] for r in records),
        "mean_fixed_skill_u": mean(r["fixed_skill_u"] for r in records),
        "mean_random_u": mean(mean(r["utilities"].values()) for r in records),
        "mean_regret": {
            "oracle": 0.0,
            "heuristic": _mean_regret("regret_heuristic"),
            "fixed_episodic": _mean_regret("regret_fixed_episodic"),
            "fixed_semantic": _mean_regret("regret_fixed_semantic"),
            "fixed_skill": _mean_regret("regret_fixed_skill"),
            "random": mean(random_regrets),
        },
        "heuristic_carrier_counts": {},
        "heuristic_unavailable_rate": mean(1.0 if r["heuristic_mapped_from_unavailable"] else 0.0 for r in records),
        "note": "No learned router. Weight/harness not in E3b matrix; unavailable heuristic choices mapped to min U.",
    }
    for r in records:
        c = r["heuristic_carrier"]
        summary["heuristic_carrier_counts"][c] = summary["heuristic_carrier_counts"].get(c, 0) + 1
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "records.json", records)
    write_json(OUT / "E4_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
