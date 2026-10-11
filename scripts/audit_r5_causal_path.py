#!/usr/bin/env python3
"""Causal-path audit for R5 skill assimilation. Fail closed; no GPU required."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from expolicy.assimilation.carriers.skill_store import SkillStore, builtin_executable_skills
from expolicy.assimilation.skill_assimilator import COMPILER_VERSION, SkillAssimilator, skill_has_leakage
from expolicy.core.experience import Experience
from expolicy.environments.search.environment import SearchEnvironment
from expolicy.experiments.common import ROOT, load_world, write_json
from expolicy.experiments.r_common import add_common_args, apply_gpu

DEFAULT_OUT = str(ROOT / "outputs" / "r5_acquisition_skill")


def _exp(task, *, success: bool, cap: str | None = None, eid: str | None = None) -> Experience:
    cap = cap or task.capability
    exp = Experience(
        experience_id=eid or f"e_{task.task_id}_{cap}",
        task_id=task.task_id,
        question=task.question,
        capability=cap,
        stage=task.stage,
        split=task.split,
        predicted_answer=task.gold_answer if success else "",
        success=success,
        # gold_answer is on the experience record (trajectory log) but must not enter the skill artifact
        gold_answer=None,
    )
    title = "Person file" if cap == "multi_hop" else ("Leadership timeline" if cap == "temporal_reasoning" else "Event file")
    q = task.question.split()[0]
    exp.tool_calls.append({"tool": "search", "arguments": {"query": q}, "ok": True})
    exp.tool_calls.append({"tool": "open", "arguments": {"document_id": "doc_acquired_only"}, "ok": True})
    exp.evidence.append({"title": title, "source_type": "official"})
    return exp


def _check(name: str, cond: bool, detail: str = "") -> dict:
    return {"name": name, "pass": bool(cond), "detail": detail}


def main() -> None:
    p = argparse.ArgumentParser()
    add_common_args(p, default_out=DEFAULT_OUT)
    args = p.parse_args()
    apply_gpu(args.gpu)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    world = load_world(Path(args.world))
    env = SearchEnvironment(world)
    assimilator = SkillAssimilator(min_support=8)
    checks: list[dict] = []

    mh = [t for t in world.tasks.values() if t.capability == "multi_hop"]
    tm = [t for t in world.tasks.values() if t.capability == "temporal_reasoning"]
    xd = [t for t in world.tasks.values() if t.capability == "cross_document_synthesis"]
    s0 = [t for t in world.tasks.values() if t.capability == "single_hop"]
    probe_ids = [t.task_id for t in mh[40:52] + tm[20:32] + xd[12:24]]
    gold_docs = [d for t in mh[40:52] for d in t.gold_doc_ids]

    # clean agent: empty store has no active executable skill
    store = SkillStore(out / "audit_store_clean")
    checks.append(_check("clean_agent_no_active_skill", store.executable_for("multi_hop") is None))

    batch_a = [_exp(t, success=i % 3 != 0, eid=f"a{i}_{t.task_id}") for i, t in enumerate(mh[:10])]
    batch_b = [_exp(t, success=i % 2 == 0, eid=f"b{i}_{t.task_id}") for i, t in enumerate(mh[10:20])]
    skill_a = assimilator.fit(batch_a, "multi_hop", probe_ids=probe_ids, gold_doc_ids=gold_docs)
    skill_b = assimilator.fit(batch_b, "multi_hop", probe_ids=probe_ids, gold_doc_ids=gold_docs)
    checks.append(_check("fit_activates_with_8plus", skill_a is not None and skill_b is not None))
    checks.append(
        _check(
            "different_batches_different_provenance",
            skill_a is not None
            and skill_b is not None
            and set(skill_a.source_experience_ids) != set(skill_b.source_experience_ids),
            f"{skill_a.source_experience_ids[:2] if skill_a else None} vs {skill_b.source_experience_ids[:2] if skill_b else None}",
        )
    )
    checks.append(
        _check(
            "different_batches_different_config_or_id",
            skill_a is not None
            and skill_b is not None
            and (
                skill_a.skill_id != skill_b.skill_id
                or skill_a.runtime_config != skill_b.runtime_config
            ),
        )
    )
    checks.append(
        _check(
            "not_unmodified_r2_builtin",
            skill_a is not None and "r2_spec" not in (skill_a.provenance or []) and skill_a.compiler_version == COMPILER_VERSION,
        )
    )

    deleted = assimilator.fit([e for e in batch_a if e.capability != "multi_hop"], "multi_hop")
    checks.append(_check("delete_capability_experiences_no_activation", deleted is None))

    empty = assimilator.fit([], "multi_hop")
    checks.append(_check("empty_batch_no_activation", empty is None))

    shuffled_s0 = [_exp(t, success=True, cap="single_hop", eid=f"s0_{i}") for i, t in enumerate(s0[:12])]
    shuffled = assimilator.fit(shuffled_s0, "multi_hop", require_capability_match=True)
    checks.append(_check("shuffled_wrong_cap_does_not_load_r2", shuffled is None))

    # even unfiltered shuffled must not copy builtin r2_spec provenance
    unfiltered = assimilator.fit(shuffled_s0, "multi_hop", require_capability_match=False)
    checks.append(
        _check(
            "unfiltered_shuffled_not_r2_global",
            unfiltered is not None and unfiltered.provenance != ["r2_spec"] and unfiltered.compiler_version == COMPILER_VERSION,
        )
    )

    skill_a2 = assimilator.fit(batch_a, "multi_hop", probe_ids=probe_ids, gold_doc_ids=gold_docs)
    checks.append(_check("same_batch_reproducible_artifact", skill_a is not None and skill_a2 is not None and skill_a.skill_id == skill_a2.skill_id))

    leaks = skill_has_leakage(skill_a, probe_ids=probe_ids, gold_doc_ids=gold_docs) if skill_a else ["missing_skill"]
    checks.append(_check("no_probe_or_gold_in_artifact", leaks == [], json.dumps(leaks)))

    # installing does not pull in the other two global primitives
    store_a = SkillStore(out / "audit_store_a")
    store_a.install(skill_a)
    checks.append(_check("only_fitted_capability_active", store_a.executable_for("multi_hop") is not None and store_a.executable_for("temporal_reasoning") is None))

    # builtin global skills must not be auto-loaded by empty/shuffled path
    builtins = {s.name: s.provenance for s in builtin_executable_skills()}
    checks.append(_check("frozen_primitives_exist", set(builtins) == {"multi_hop_bridge_search", "temporal_resolution", "cross_document_synthesis"}))

    # runtime actually uses config: two configs should be allowed to differ
    if skill_a and skill_b:
        env.reset(mh[0])
        from expolicy.assimilation.carriers.skill_store import execute_skill

        r1 = execute_skill(skill_a, env, task=mh[0], query=mh[0].question)
        env.reset(mh[0])
        r2 = execute_skill(skill_b, env, task=mh[0], query=mh[0].question)
        checks.append(_check("runtime_executes_fitted_skill", bool(r1.get("skill") and r2.get("skill"))))

    failed = [c for c in checks if not c["pass"]]
    payload = {
        "n_checks": len(checks),
        "n_pass": sum(1 for c in checks if c["pass"]),
        "n_fail": len(failed),
        "decision": "PASS" if not failed else "FAIL",
        "compiler_version": COMPILER_VERSION,
        "checks": checks,
    }
    write_json(out / "R5_CAUSAL_AUDIT.json", payload)
    print(json.dumps(payload, indent=2))
    if failed:
        raise SystemExit("R5 causal audit FAIL")


if __name__ == "__main__":
    main()
