"""Observable features and answer extraction. Gold answers are not inputs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
_DIRECTED = re.compile(r"directed by ([A-Z][A-Za-z0-9]+(?: [A-Z][A-Za-z0-9]+)*)")
_CAPITAL = re.compile(r"capital of .+? is ([A-Z][A-Za-z0-9]+(?: [A-Z][A-Za-z0-9]+)*)", re.IGNORECASE)
_COMPARISON_SPLIT = re.compile(r",\s*(.+?)\s+or\s+(.+?)\??$", re.IGNORECASE)
_FILM = re.compile(r"(?:film|movie)\s+(.+?)\??$", re.IGNORECASE)
_CAPITAL_ENTITY = re.compile(r"capital of\s+(.+?)\??$", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+")


def clean_token(text: str) -> str:
    return text.strip().strip("?.,;:\"'")


def parse_entities(question: str) -> list[str]:
    match = _COMPARISON_SPLIT.search(question.strip())
    if match:
        return [clean_token(match.group(1)), clean_token(match.group(2))]
    match = _FILM.search(question.strip())
    if match:
        return [clean_token(match.group(1))]
    match = _CAPITAL_ENTITY.search(question.strip())
    if match:
        return [clean_token(match.group(1))]
    return []


def infer_attribute_kind(question: str) -> str:
    text = question.casefold()
    if "director" in text or "directed" in text:
        return "director"
    if "capital" in text:
        return "location"
    if any(word in text for word in ("release", "released", "came out", "film")):
        return "release_date"
    return "other"


def attribute_phrase(kind: str) -> str:
    return {
        "release_date": "release date",
        "director": "director",
        "location": "capital",
        "other": "fact",
    }.get(kind, "fact")


def has_comparison_cue(question: str) -> int:
    text = question.casefold()
    cues = ("which", " or ", "first", "earlier", "later", "older", "younger", "same")
    return int(any(cue in text for cue in cues))


def comparison_direction(question: str) -> str:
    text = question.casefold()
    if any(word in text for word in ("later", "last", "youngest", "after")):
        return "later"
    return "earlier"


def observable_features(
    question: str,
    *,
    evidence: Mapping[str, str],
    search_count: int,
    step_count: int,
    last_observation: str,
) -> dict[str, Any]:
    observation = last_observation or ""
    return {
        "n_entities": len(parse_entities(question)),
        "has_comparison_cue": has_comparison_cue(question),
        "attribute_kind": infer_attribute_kind(question),
        "search_count": int(search_count),
        "last_observation_empty": int(search_count > 0 and not observation.strip()),
        "evidence_filled": len(evidence),
        "has_directed_by_span": int(_DIRECTED.search(observation) is not None),
        "step_count": int(step_count),
    }


def atom_holds(features: Mapping[str, Any], atom: Mapping[str, Any]) -> bool:
    feature = str(atom.get("feature") or "")
    if feature not in features:
        return False
    left = features[feature]
    op = str(atom.get("op") or "")
    right = atom.get("value")
    if op == "eq":
        return left == right
    if op == "neq":
        return left != right
    if op == "gte":
        return float(left) >= float(right)
    if op == "lte":
        return float(left) <= float(right)
    if op == "in":
        return left in list(right or [])
    return False


def trigger_matches(trigger: Mapping[str, Any], features: Mapping[str, Any], *, max_atoms: int = 3) -> bool:
    atoms = list(trigger.get("all") or [])
    if not atoms or len(atoms) > max_atoms:
        return False
    return all(atom_holds(features, atom) for atom in atoms)


def extract_slot(observation: str) -> str:
    directed = _DIRECTED.search(observation or "")
    if directed:
        return directed.group(1)
    years = _YEAR.findall(observation or "")
    if years:
        return years[0]
    return ""


def extract_base_answer(observation: str) -> str:
    """What the frozen base agent can read off one observation. Years are not answers."""
    capital = _CAPITAL.search(observation or "")
    if capital:
        return capital.group(1)
    directed = _DIRECTED.search(observation or "")
    if directed:
        return directed.group(1)
    return ""


def redact_observation(observation: str) -> str:
    text = _DIRECTED.sub("directed by VALUE", observation or "")
    text = _CAPITAL.sub("capital of PLACE is VALUE", text)
    text = _YEAR.sub("YEAR", text)
    return text


def contains_term(query: str, term: str) -> bool:
    words = set(_WORD.findall(query.casefold()))
    return term.casefold() in words


def is_year(value: str) -> bool:
    return bool(re.fullmatch(r"\d{4}", value or ""))


def exact_match(prediction: str, gold: str) -> bool:
    return prediction.strip().casefold() == gold.strip().casefold()
