"""Local MiniLM embedder using transformers (no sentence-transformers required)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from expolicy.core.errors import BackendUnavailableError

DEFAULT_EMBEDDER = "/data/ppnm/models/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_path: str = DEFAULT_EMBEDDER, device: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise BackendUnavailableError(
                "Embedder requires torch and transformers to load MiniLM."
            ) from exc
        path = Path(model_path)
        if not path.exists():
            raise BackendUnavailableError(
                f"embedding model missing: {model_path}. Place MiniLM at that path."
            )
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModel.from_pretrained(str(path))
        self.model.to(self.device)
        self.model.eval()
        self.dim = int(self.model.config.hidden_size)
        self.name = str(path)

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        torch = self.torch
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start : start + batch_size])
            if not chunk:
                continue
            encoded = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                output = self.model(**encoded)
                hidden = output.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy())
        if not vectors:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.concatenate(vectors, axis=0).astype(np.float32)


class VectorIndex:
    """Cosine index. Uses FAISS when installed, otherwise NumPy."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.ids: list[str] = []
        self.vectors: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._faiss = None
        try:
            import faiss  # type: ignore

            self._faiss_mod = faiss
            self._faiss = faiss.IndexFlatIP(dim)
        except ImportError:
            self._faiss_mod = None

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if vectors.size == 0:
            return
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape}")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-8)
        vectors = (vectors / norms).astype(np.float32)
        self.ids.extend(ids)
        self.vectors = np.concatenate([self.vectors, vectors], axis=0) if self.vectors.size else vectors
        if self._faiss is not None:
            self._faiss.add(vectors)

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        q = query.astype(np.float32).reshape(1, -1)
        q = q / np.linalg.norm(q, axis=1, keepdims=True).clip(min=1e-8)
        k = min(k, len(self.ids))
        if self._faiss is not None:
            scores, idxs = self._faiss.search(q, k)
            return [(self.ids[int(i)], float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]
        scores = (self.vectors @ q.T).ravel()
        top = np.argsort(-scores)[:k]
        return [(self.ids[int(i)], float(scores[int(i)])) for i in top]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        (path / "ids.json").write_text(__import__("json").dumps(self.ids), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, dim: int) -> VectorIndex:
        import json

        index = cls(dim)
        vec_path = path / "vectors.npy"
        ids_path = path / "ids.json"
        if vec_path.exists() and ids_path.exists():
            vectors = np.load(vec_path)
            ids = json.loads(ids_path.read_text(encoding="utf-8"))
            index.add(ids, vectors)
        return index
