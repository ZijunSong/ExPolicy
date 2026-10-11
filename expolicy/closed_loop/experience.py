"""Experience batch summary for assimilation observations. No future labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from expolicy.core.experience import Experience


@dataclass
class ExperienceBatchSummary:
    capability: str
    n: int
    success_rate: float
    failure_rate: float
    mean_reward: float
    mean_searches: float
    mean_turns: float
    repeat_query_rate: float
    premature_answer_rate: float
    retrieval_miss_rate: float
    answer_synthesis_failure_rate: float
    unique_query_ratio: float
    evidence_completion_rate: float
    token_cost: int
    tool_cost: int
    task_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperienceBatchSummary":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def summarize_batch(experiences: Sequence[Experience], *, capability: str) -> ExperienceBatchSummary:
    n = len(experiences)
    if n == 0:
        return ExperienceBatchSummary(
            capability=capability,
            n=0,
            success_rate=0.0,
            failure_rate=1.0,
            mean_reward=0.0,
            mean_searches=0.0,
            mean_turns=0.0,
            repeat_query_rate=0.0,
            premature_answer_rate=0.0,
            retrieval_miss_rate=0.0,
            answer_synthesis_failure_rate=0.0,
            unique_query_ratio=0.0,
            evidence_completion_rate=0.0,
            token_cost=0,
            tool_cost=0,
        )
    successes = [1.0 if e.success else 0.0 for e in experiences]
    n_search, n_turn, repeats, premature, miss, synth_fail, uniq_ratio, evidence = [], [], [], [], [], [], [], []
    tokens = 0
    tools = 0
    for exp in experiences:
        searches = [str((c.get("arguments") or {}).get("query") or "") for c in exp.tool_calls if c.get("tool") == "search"]
        opens = [c for c in exp.tool_calls if c.get("tool") == "open"]
        n_search.append(float(len(searches)))
        n_turn.append(float(len(exp.actions)))
        qnorm = [q.strip().lower() for q in searches if q.strip()]
        uniq = len(set(qnorm))
        uniq_ratio.append(uniq / max(1, len(qnorm)))
        repeats.append(1.0 if len(qnorm) >= 2 and uniq < len(qnorm) else 0.0)
        answered_early = False
        opened_before_answer = False
        saw_open = False
        for act in exp.actions:
            if act.get("tool") == "open":
                saw_open = True
            if act.get("tool") == "answer":
                answered_early = not saw_open
                opened_before_answer = saw_open
                break
        premature.append(1.0 if answered_early else 0.0)
        miss.append(1.0 if not opens else 0.0)
        pred = (exp.predicted_answer or "").strip()
        synth_fail.append(1.0 if not pred else 0.0)
        evidence.append(1.0 if opened_before_answer or len(opens) >= 2 else 0.0)
        tokens += int(exp.cost.tokens or 0)
        tools += int(exp.cost.tool_calls or 0)
    return ExperienceBatchSummary(
        capability=capability,
        n=n,
        success_rate=_mean(successes),
        failure_rate=1.0 - _mean(successes),
        mean_reward=_mean([float(e.reward or 0.0) for e in experiences]),
        mean_searches=_mean(n_search),
        mean_turns=_mean(n_turn),
        repeat_query_rate=_mean(repeats),
        premature_answer_rate=_mean(premature),
        retrieval_miss_rate=_mean(miss),
        answer_synthesis_failure_rate=_mean(synth_fail),
        unique_query_ratio=_mean(uniq_ratio),
        evidence_completion_rate=_mean(evidence),
        token_cost=tokens,
        tool_cost=tools,
        task_ids=[str(e.task_id) for e in experiences],
    )
