"""Frozen deterministic capability mapping for public multi-hop search.

Primary labels use dataset-public metadata only. Future gold answers / supporting
facts are never used to infer acquisition-side capability.
"""

from __future__ import annotations

from typing import Any

C1 = "bridge_decomposition"
C2 = "comparison_alignment"
C3 = "long_chain_search"
PUBLIC_CAPS = (C1, C2, C3)
UNMAPPED = "unmapped"
AMBIGUOUS = "ambiguous"

MAPPING_RULES = """
C1 bridge_decomposition:
  - HotpotQA type == bridge (or missing type on a 2-hop non-comparison item)
  - 2Wiki type in {bridge, inference, compositional} AND hops < 3
  - MuSiQue 2-hop items
  - Bamboogle default (probe only)

C2 comparison_alignment:
  - HotpotQA type == comparison
  - 2Wiki type in {comparison, bridge_comparison}

C3 long_chain_search:
  - MuSiQue hops >= 3 (id prefix 3hop/4hop or decomposition length)
  - 2Wiki type in {bridge, inference, compositional} AND hops >= 3

Unmapped: empty question, empty answer (except bamboogle test-without-answer is dropped).
Ambiguous: 2Wiki type unknown AND hops == 2 → C1 (logged as defaulted, not ambiguous).
Dataset name is never used as a capability id.
"""


def map_capability(record: dict[str, Any]) -> dict[str, Any]:
    dataset = str(record.get("dataset") or "")
    qtype = str(record.get("type") or "").strip().lower()
    hops = int(record.get("hops") or 2)
    question = str(record.get("question") or "").strip()
    answer = str(record.get("gold_answer") or "").strip()
    reason = ""
    cap = UNMAPPED
    ambiguous = False

    if not question:
        return {"capability": UNMAPPED, "reason": "empty_question", "ambiguous": False, "unmapped": True}

    if dataset == "hotpotqa":
        if qtype == "comparison":
            cap, reason = C2, "hotpotqa.type=comparison"
        elif qtype == "bridge" or qtype == "":
            cap, reason = C1, "hotpotqa.type=bridge_or_default"
        else:
            cap, reason, ambiguous = C1, f"hotpotqa.type={qtype}->default_c1", True
    elif dataset == "2wiki":
        if qtype in {"comparison", "bridge_comparison"}:
            cap, reason = C2, f"2wiki.type={qtype or 'comparison'}"
        elif qtype in {"bridge", "inference", "compositional"}:
            if hops >= 3:
                cap, reason = C3, f"2wiki.type={qtype};hops={hops}"
            else:
                cap, reason = C1, f"2wiki.type={qtype};hops={hops}"
        elif hops >= 3:
            cap, reason, ambiguous = C3, f"2wiki.type={qtype or 'unknown'};hops={hops}", True
        else:
            cap, reason, ambiguous = C1, f"2wiki.type={qtype or 'unknown'}->default_c1", True
    elif dataset == "musique":
        if hops >= 3:
            cap, reason = C3, f"musique.hops={hops}"
        else:
            cap, reason = C1, f"musique.hops={hops}"
    elif dataset == "bamboogle":
        cap, reason = C1, "bamboogle.default_bridge"
    else:
        cap, reason = UNMAPPED, f"unknown_dataset={dataset}"

    if dataset != "bamboogle" and not answer:
        return {
            "capability": UNMAPPED,
            "reason": "empty_answer",
            "ambiguous": False,
            "unmapped": True,
        }
    return {
        "capability": cap,
        "reason": reason,
        "ambiguous": ambiguous,
        "unmapped": cap == UNMAPPED,
    }


def apply_mapping(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for rec in records:
        mapped = dict(rec)
        info = map_capability(rec)
        mapped.update(info)
        out.append(mapped)
    return out
