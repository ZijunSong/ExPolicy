"""Teacher / correction rollouts that produce environment-verified successful trajectories."""

from __future__ import annotations

from expolicy.agents.llm_agent import _normalize_answer, _soft_match
from expolicy.assimilation.carriers.skill_store import (
    Skill,
    run_skill_on_task,
    skill_a_multihop,
    skill_b_temporal,
    skill_c_crossdoc,
)
from expolicy.core.experience import Experience
from expolicy.environments.search.environment import SearchEnvironment
from expolicy.environments.search.world import Task, World


def skill_for_task(task: Task) -> Skill | None:
    mapping = {
        "multi_hop": skill_a_multihop,
        "query_decomposition": skill_a_multihop,
        "temporal_reasoning": skill_b_temporal,
        "cross_document_synthesis": skill_c_crossdoc,
    }
    fn = mapping.get(task.capability or "")
    return fn() if fn else None


def teacher_rollout(env: SearchEnvironment, task: Task) -> Experience:
    """Run the executable skill. If it fails, search then open gold docs and re-extract.

    Gold ids are last-resort recovery after a real search; the answer is still read from
    opened document text and checked against the environment gold label.
    """
    skill = skill_for_task(task)
    if skill is None:
        return Experience(
            task_id=task.task_id,
            question=task.question,
            gold_answer=task.gold_answer,
            capability=task.capability,
            predicted_answer="",
            success=False,
        )
    exp = run_skill_on_task(skill, env, task)
    if exp.success:
        return exp
    env.reset(task)
    env.step("search", {"query": task.question})
    opened = []
    for did in task.gold_doc_ids:
        result = env.step("open", {"document_id": did})
        if result.ok:
            opened.append(result.payload)
            exp.tool_calls.append({"tool": "open", "arguments": {"document_id": did}, "ok": True})
            exp.observations.append({"tool": "open", "payload": result.payload})
            exp.evidence.append(
                {
                    "document_id": did,
                    "title": result.payload.get("title"),
                    "source_type": result.payload.get("source_type"),
                }
            )
    from expolicy.assimilation.carriers.skill_store import (
        _extract_event_org_lead,
        _synthesize_crossdoc,
        _synthesize_multihop,
        _synthesize_temporal,
    )
    from expolicy.experiments.patterns import mentioned_entities

    ents = mentioned_entities(task.question, env.world)
    year = (ents.get("years") or [None])[0]
    if task.capability == "temporal_reasoning":
        answer = _synthesize_temporal(task.question, opened, year)
    elif task.capability == "cross_document_synthesis":
        org, lead = _extract_event_org_lead(opened)
        answer = _synthesize_crossdoc(org, lead)
    else:
        field = "ceo" if "chief executive" in task.question.lower() else "headquarters"
        if "sector" in task.question.lower():
            field = "sector"
        answer = _synthesize_multihop(
            task.question,
            opened,
            {
                "person": (ents.get("people") or [None])[0],
                "org": (ents.get("orgs") or [None])[0],
                "target_field": field,
            },
        )
    if not answer:
        return exp
    exp.predicted_answer = answer
    success = _normalize_answer(answer) == _normalize_answer(task.gold_answer) or _soft_match(
        answer, task.gold_answer, task.aliases
    )
    exp.success = success
    exp.reward = 1.0 if success else 0.0
    exp.parent_experience_id = exp.experience_id
    exp.branch_origin = {"kind": "teacher_correction", "parent_task": task.task_id}
    exp.outcome["predicted"] = answer
    exp.outcome["success"] = success
    exp.outcome["corrected"] = True
    return exp


def collect_verified_successes(
    world: World,
    tasks: list[Task],
    *,
    existing: list[Experience] | None = None,
    min_n: int = 32,
) -> list[Experience]:
    env = SearchEnvironment(world)
    verified: list[Experience] = []
    seen: set[str] = set()
    for exp in existing or []:
        if exp.success and exp.task_id not in seen:
            verified.append(exp)
            seen.add(exp.task_id)
    for task in tasks:
        if task.task_id in seen:
            continue
        rolled = teacher_rollout(env, task)
        if rolled.success:
            verified.append(rolled)
            seen.add(task.task_id)
        if len(verified) >= min_n:
            break
    return verified
