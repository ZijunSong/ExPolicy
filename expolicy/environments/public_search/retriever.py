"""HTTP client for the local Wikipedia E5 retriever (Search-R1 compatible)."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urljoin

import httpx

DEFAULT_RETRIEVER_URL = os.environ.get("EXPOLICY_RETRIEVER_URL", "http://127.0.0.1:8000")


class RetrieverClient:
    def __init__(self, base_url: str = DEFAULT_RETRIEVER_URL, *, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def retrieve(self, queries: list[str], *, topk: int = 5, return_scores: bool = True) -> list[list[dict[str, Any]]]:
        url = urljoin(self.base_url + "/", "retrieve")
        payload = {"queries": queries, "topk": topk, "return_scores": return_scores}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        rows = data.get("result") or []
        out: list[list[dict[str, Any]]] = []
        for group in rows:
            hits: list[dict[str, Any]] = []
            for item in group or []:
                if isinstance(item, dict) and "document" in item:
                    doc = dict(item.get("document") or {})
                    doc["score"] = item.get("score")
                    hits.append(_normalize_doc(doc))
                elif isinstance(item, dict):
                    hits.append(_normalize_doc(item))
            out.append(hits)
        return out

    def search(self, query: str, *, topk: int = 5) -> list[dict[str, Any]]:
        groups = self.retrieve([query], topk=topk)
        return groups[0] if groups else []

    def open_id(self, document_id: str) -> dict[str, Any] | None:
        url = urljoin(self.base_url + "/", "open")
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json={"document_id": document_id})
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                return _normalize_doc(resp.json().get("document") or resp.json())
        except Exception:
            return None

    def health(self) -> dict[str, Any]:
        url = urljoin(self.base_url + "/", "health")
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.json()


def _normalize_doc(doc: dict[str, Any]) -> dict[str, Any]:
    contents = str(doc.get("contents") or "")
    title = str(doc.get("title") or "")
    text = str(doc.get("text") or "")
    if contents and not title:
        lines = contents.split("\n", 1)
        title = lines[0].strip().strip('"')
        text = lines[1] if len(lines) > 1 else contents
    elif contents and not text:
        lines = contents.split("\n", 1)
        text = lines[1] if len(lines) > 1 else contents
    doc_id = str(doc.get("id") or doc.get("document_id") or "")
    return {
        "document_id": doc_id,
        "id": doc_id,
        "title": title,
        "text": text,
        "contents": contents or f"{title}\n{text}".strip(),
        "score": float(doc.get("score") or 0.0),
        "source_type": "wikipedia",
        "reliability": 0.9,
        "year": 2018,
        "snippet": (text or contents)[:280],
    }
