"""Length-matched controls. A ratio above 0.20 blocks evaluation."""

from __future__ import annotations

import itertools
from typing import Any, Callable, Sequence

from expolicy.paper_d.confirm_protocol import PROTOCOL
from expolicy.paper_d.content_controls import empty_leaks_source
from expolicy.paper_d.content_tasks import jaccard, token_set
from expolicy.r9_lab.assimilators.episodic import format_episodic_injection, retrieve_episodic
from expolicy.r9_lab.state.persistent_state import EpisodicRecord, PersistentState


def memory_text(records: Sequence[dict[str, Any]], instruction: str) -> str:
    state = PersistentState(root="")
    state.episodic = [
        EpisodicRecord(
            task_id=str(record["task_id"]),
            env="os",
            instruction=str(record.get("instruction") or ""),
            trajectory=list(record.get("trajectory") or []),
            success=True,
        )
        for record in records
    ]
    state.committed_task_ids = [record.task_id for record in state.episodic]
    return format_episodic_injection(retrieve_episodic(state, instruction))


def length_ratio(real_tokens: int, wrong_tokens: int) -> float | None:
    if real_tokens <= 0:
        return None
    return abs(wrong_tokens - real_tokens) / real_tokens


def _eligible(donor: dict[str, Any], real_records: Sequence[dict[str, Any]]) -> bool:
    real_skills = {skill for record in real_records for skill in (record.get("skill_list") or [])}
    donor_skills = set(donor.get("skill_list") or [])
    if jaccard(donor_skills, real_skills) > float(PROTOCOL["skill_distance_max"]):
        return False
    limit = float(PROTOCOL["near_copy_jaccard"])
    return all(
        jaccard(token_set(str(donor.get("instruction") or "")), token_set(str(record.get("instruction") or ""))) < limit
        for record in real_records
    )


def choose_wrong_bundle(
    real_records: Sequence[dict[str, Any]],
    donors: Sequence[dict[str, Any]],
    eval_instructions: Sequence[str],
    counter: Callable[[str], int],
) -> dict[str, Any]:
    """Pick one donor bundle by frozen length and skill rules. No evaluation scores."""
    if not real_records:
        return {"status": "EMPTY_UPDATE_BATCH", "donors": [], "ratios": []}
    k = len(real_records)
    eligible = [donor for donor in donors if _eligible(donor, real_records)]
    if len(eligible) < k:
        return {"status": "insufficient_donors", "donors": [], "ratios": [], "n_eligible": len(eligible)}
    real_probe = str(real_records[0].get("instruction") or "task")
    real_chunks = [
        counter(memory_text([record], real_probe))
        for record in real_records
    ]
    target = sum(real_chunks) / k
    ranked = sorted(
        enumerate(eligible),
        key=lambda item: (abs(counter(memory_text([item[1]], real_probe)) - target), item[0]),
    )
    closest = [item[1] for item in ranked[:12]]
    ranked_combos = []
    for combo in itertools.combinations(closest, k):
        wrong_tokens = counter(memory_text(list(combo), real_probe))
        real_tokens = counter(memory_text(list(real_records), real_probe))
        ratio = length_ratio(real_tokens, wrong_tokens)
        ranked_combos.append((ratio if ratio is not None else 99.0, combo))
    ranked_combos.sort(key=lambda item: item[0])
    tolerance = float(PROTOCOL["length_tolerance"])
    for _proxy, combo in ranked_combos[:20]:
        ratios = []
        failed = False
        for instruction in eval_instructions:
            real_tokens = counter(memory_text(list(real_records), instruction))
            wrong_tokens = counter(memory_text(list(combo), instruction))
            ratio = length_ratio(real_tokens, wrong_tokens)
            ratios.append({"instruction_hash_prefix": instruction[:24], "real_tokens": real_tokens, "wrong_tokens": wrong_tokens, "ratio": ratio})
            if ratio is None or ratio > tolerance:
                failed = True
                break
        if not failed:
            return {"status": "matched", "donors": list(combo), "ratios": ratios, "n_eligible": len(eligible)}
    return {"status": "length_gate_failed", "donors": [], "ratios": [], "n_eligible": len(eligible)}


def empty_has_source_text(real_records: Sequence[dict[str, Any]], instruction: str) -> list[str]:
    neutralized = []
    for record in real_records:
        neutralized.append(
            {
                **record,
                "instruction": str(PROTOCOL["neutral_instruction"]),
                "trajectory": [{"agent": str(PROTOCOL["neutral_action"])}],
            }
        )
    return empty_leaks_source(memory_text(neutralized, instruction), real_records)


def load_tokenizer_counter() -> Callable[[str], int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(PROTOCOL["tokenizer_path"]), trust_remote_code=True)

    def counter(text: str) -> int:
        return len(tokenizer.encode(text or "", add_special_tokens=False))

    return counter
