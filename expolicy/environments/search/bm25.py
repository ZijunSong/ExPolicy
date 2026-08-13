"""BM25 retriever over the Helion Archive corpus."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from expolicy.environments.search.world import Document, World

_TOKEN = re.compile(r"[A-Za-z0-9_\-]+")


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in _TOKEN.findall(text)]


@dataclass
class SearchHit:
    document_id: str
    title: str
    score: float
    rank: int
    snippet: str
    source_type: str
    reliability: float
    year: int


class BM25Index:
    def __init__(self, documents: list[Document]):
        self.documents = list(documents)
        self._by_id = {doc.document_id: doc for doc in self.documents}
        self._tokens = [tokenize(f"{doc.title} {doc.text}") for doc in self.documents]
        if not self._tokens:
            raise ValueError("BM25Index requires at least one document")
        self._bm25 = BM25Okapi(self._tokens)

    @classmethod
    def from_world(cls, world: World) -> BM25Index:
        return cls(list(world.documents.values()))

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), self.documents[i].document_id))
        hits: list[SearchHit] = []
        for rank, idx in enumerate(ranked[:top_k], start=1):
            score = float(scores[idx])
            if score <= 0:
                continue
            doc = self.documents[idx]
            snippet = doc.text[:280]
            hits.append(
                SearchHit(
                    document_id=doc.document_id,
                    title=doc.title,
                    score=score,
                    rank=rank,
                    snippet=snippet,
                    source_type=doc.source_type,
                    reliability=doc.reliability,
                    year=doc.year,
                )
            )
        return hits

    def open(self, document_id: str) -> Document:
        if document_id not in self._by_id:
            raise KeyError(f"unknown document_id: {document_id}")
        return self._by_id[document_id]

    def find(self, document_id: str, pattern: str, window: int = 180) -> list[dict[str, str | int]]:
        doc = self.open(document_id)
        text = doc.text
        matches: list[dict[str, str | int]] = []
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            regex = re.compile(re.escape(pattern), re.IGNORECASE)
        for match in regex.finditer(text):
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            matches.append(
                {
                    "document_id": document_id,
                    "span": text[start:end],
                    "start": match.start(),
                    "end": match.end(),
                }
            )
            if len(matches) >= 8:
                break
        return matches
