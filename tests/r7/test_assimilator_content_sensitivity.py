"""Different experience batches must produce different SkillOpt artifacts (content sensitivity)."""

from __future__ import annotations

from expolicy.core.experience import Experience
from expolicy.r7.interventional_skills import skill_s_bridge
from expolicy.r7.skillopt import analyze_batch, apply_edit, propose_edits, skill_hash


def _batch(tag: str, questions: list[str]) -> list[Experience]:
    rows = []
    for i, q in enumerate(questions):
        rows.append(
            Experience(
                task_id=f"{tag}:{i}",
                question=q,
                capability="bridge_decomposition",
                success=False,
                predicted_answer="",
            )
        )
    return rows


def test_different_batches_yield_different_skill_hashes() -> None:
    years = _batch(
        "year",
        [
            "In which year was the Eiffel Tower completed?",
            "What year did the French Revolution begin?",
            "When was the Battle of Hastings?",
            "What year did World War I start?",
            "When was the Magna Carta sealed?",
            "What year was the UN founded?",
            "When did the Berlin Wall fall?",
            "What year did Apollo 11 land?",
        ],
    )
    films = _batch(
        "film",
        [
            "Who directed the film Inception?",
            "Which film won Best Picture in 1994?",
            "Who starred in the movie Casablanca?",
            "What film featured the character Vito Corleone?",
            "Who directed the movie Psycho?",
            "Which film is set in the Overlook Hotel?",
            "Who composed the score for Jaws?",
            "What movie features the line I'll be back?",
        ],
    )
    s0 = skill_s_bridge()
    e0 = propose_edits(s0, analyze_batch(years, "bridge_decomposition"), max_edits=4)
    e1 = propose_edits(s0, analyze_batch(films, "bridge_decomposition"), max_edits=4)
    assert e0 and e1
    h0 = skill_hash(apply_edit(s0, e0[0])) if e0 else skill_hash(s0)
    h1 = skill_hash(apply_edit(s0, e1[0])) if e1 else skill_hash(s0)
    assert h0 != h1
    a0 = analyze_batch(years, "bridge_decomposition")
    a1 = analyze_batch(films, "bridge_decomposition")
    assert a0["query_suffixes"] != a1["query_suffixes"] or a0["majority_tokens"] != a1["majority_tokens"]


def test_same_batch_is_deterministic() -> None:
    qs = ["Who founded Acme Corporation in Detroit?", "Who founded Acme Corporation in Detroit?"] * 4
    b = _batch("same", qs)
    s = skill_s_bridge()
    e1 = propose_edits(s, analyze_batch(b, "bridge_decomposition"), max_edits=4)
    e2 = propose_edits(s, analyze_batch(b, "bridge_decomposition"), max_edits=4)
    assert e1 == e2
