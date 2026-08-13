"""Active experience generation: hard variants, real branch replay, failure-seeking."""

from __future__ import annotations

import copy
import random
from typing import Any

from expolicy.agents.llm_agent import LLMSearchAgent
from expolicy.core.capability import CapabilityProfile
from expolicy.core.experience import Experience
from expolicy.environments.search.environment import SearchEnvironment
from expolicy.environments.search.world import Task, World


class VariantValidator:
    def __init__(self, world: World) -> None:
        self.world = world

    def valid(self, task: Task) -> tuple[bool, str]:
        if not task.question.strip() or not task.gold_answer.strip():
            return False, "empty"
        if task.gold_answer.lower() in {"none", "unknown", "n/a"}:
            return False, "degenerate_answer"
        # gold docs must exist
        for doc_id in task.gold_doc_ids:
            if doc_id not in self.world.documents:
                return False, f"missing_gold_doc:{doc_id}"
        return True, "ok"


class HardVariantGenerator:
    def __init__(self, world: World, seed: int = 0) -> None:
        self.world = world
        self.rng = random.Random(seed)
        self.validator = VariantValidator(world)

    def generate(self, base: Task, n: int = 3) -> list[Task]:
        variants: list[Task] = []
        ops = [
            self._add_distractor_clause,
            self._add_time_condition,
            self._rephrase_entity,
        ]
        for i in range(n):
            task = copy.deepcopy(base)
            task.task_id = f"{base.task_id}::hard{i}"
            task.question = ops[i % len(ops)](base)
            task.metadata = dict(base.metadata)
            task.metadata["parent_task_id"] = base.task_id
            task.metadata["variant"] = "hard"
            ok, reason = self.validator.valid(task)
            if ok:
                variants.append(task)
            else:
                task.metadata["rejected"] = reason
        return variants

    def _add_distractor_clause(self, base: Task) -> str:
        city = self.rng.choice(list({d.city for d in self.world.documents.values()}))
        return f"{base.question} Ignore unrelated reports from {city} blogs."

    def _add_time_condition(self, base: Task) -> str:
        return f"Using only sources dated 2018 or later: {base.question}"

    def _rephrase_entity(self, base: Task) -> str:
        return f"From the Helion Archive records, {base.question[0].lower() + base.question[1:]}"


class BranchExplorer:
    """Restart from a real trajectory prefix and execute alternative queries."""

    def __init__(self, agent: LLMSearchAgent) -> None:
        self.agent = agent

    def explore(self, experience: Experience, task: Task, n_branches: int = 3) -> list[Experience]:
        branches: list[Experience] = []
        alt_queries = _alternative_queries(experience, n_branches)
        original_max = self.agent.harness.max_steps
        for i, query in enumerate(alt_queries):
            # Real execution: force a first search with an alternate query by wrapping the question.
            branched_task = Task(
                task_id=f"{task.task_id}::branch{i}",
                question=f"{task.question} (start by searching: {query})",
                gold_answer=task.gold_answer,
                capability=task.capability,
                stage=task.stage,
                gold_doc_ids=task.gold_doc_ids,
                distractor_doc_ids=task.distractor_doc_ids,
                hops=task.hops,
                split=task.split,
                aliases=task.aliases,
                metadata={"parent_experience_id": experience.experience_id, "branch_query": query},
            )
            self.agent.harness.max_steps = min(original_max, 6)
            child = self.agent.run_task(branched_task)
            child.parent_experience_id = experience.experience_id
            child.branch_origin = {"parent": experience.experience_id, "query": query, "index": i}
            branches.append(child)
        self.agent.harness.max_steps = original_max
        return branches


class FailureSeekingGenerator:
    def __init__(self, world: World, seed: int = 0) -> None:
        self.world = world
        self.rng = random.Random(seed)
        self.hard = HardVariantGenerator(world, seed)

    def generate(self, profile: CapabilityProfile, pool: list[Task], n: int = 4) -> list[Task]:
        frontier = profile.frontier_names() or list(profile.capability_scores.keys())
        candidates = [t for t in pool if t.capability in frontier]
        self.rng.shuffle(candidates)
        out: list[Task] = []
        for base in candidates[:n]:
            out.extend(self.hard.generate(base, n=1))
        return out


def _alternative_queries(experience: Experience, n: int) -> list[str]:
    base = experience.question
    tokens = base.replace("?", "").split()
    entity = " ".join(tokens[:4]) if tokens else base
    alts = [
        entity,
        f"{entity} official registry",
        f"{experience.capability} {entity}",
        f"{base} encyclopedia",
    ]
    return alts[:n]
