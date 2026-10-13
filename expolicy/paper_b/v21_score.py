"""HotpotQA official EM and token F1. This is the only primary success label for B-v2.1."""

from __future__ import annotations

import re
import string
from collections import Counter

SCORER_VERSION = "hotpot_evaluate_v1"
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_YES_NO = frozenset({"yes", "no", "noanswer"})


def normalize_answer(text: str) -> str:
    lowered = (text or "").lower()
    without_punc = "".join(ch for ch in lowered if ch not in set(string.punctuation))
    without_articles = _ARTICLES.sub(" ", without_punc)
    return " ".join(without_articles.split())


def official_em(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def official_f1(prediction: str, gold: str) -> float:
    pred = normalize_answer(prediction)
    truth = normalize_answer(gold)
    if pred in _YES_NO and pred != truth:
        return 0.0
    if truth in _YES_NO and pred != truth:
        return 0.0
    pred_tokens = pred.split()
    gold_tokens = truth.split()
    if not pred_tokens or not gold_tokens:
        return float(pred == truth)
    common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer(prediction: str, gold: str, aliases: tuple[str, ...] = ()) -> dict[str, float]:
    candidates = [gold, *aliases]
    em = max(official_em(prediction, item) for item in candidates)
    f1 = max(official_f1(prediction, item) for item in candidates)
    return {"official_em": em, "official_f1": f1}
